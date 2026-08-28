"""The verification gate.

Verification is a *pipeline of independent checks*, not a branching ladder. Each
check is a pure function `(VerifyContext) -> Failure | None`, registered in
`CHECKS` with the routes it applies to and whether it costs an LLM call. `verify`
runs the applicable ones in order and stops at the first failure; the heal action
is then a table lookup, not another branch.

Two invariants are carried by the registry itself rather than by comments:

- **Cheap before expensive.** `CHECKS` is ordered deterministic-first, so a
  malformed citation or a fabricated number costs zero LLM calls. `_check_order`
  asserts this at import time -- reordering the table cannot silently make
  verification more expensive.
- **Route applicability is data.** Which checks a route runs lives in each
  entry's `routes`, instead of being scattered across `if route == ...` guards.

Adding a check is one function plus one registry line.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from meridian import config
from meridian.evidence import cap_pool, format_evidence, parse_markers, strip_markers
from meridian.ingest import embed_texts
from meridian.logging_config import get_logger
from meridian.nodes.llm import chat
from meridian.state import RAGState

log = get_logger("verify")

# `None` means "every route", deliberately rather than an enumeration of the two
# routes that reach verify today. A check restricted to a listed set silently
# stops running if it ever sees an unlisted route -- verification failing *open*
# on an unexpected value is the wrong direction to fail.
_ALL_ROUTES = None
_HARD_ONLY = frozenset({config.ROUTE_HARD})


# ------------------------------------------------------------------- the types

@dataclass(frozen=True)
class VerifyContext:
    """Everything the checks may read, computed once per verification."""

    question: str
    answer: str
    route: str
    sub_questions: tuple[str, ...]
    pool: list[dict]

    @property
    def n_evidence(self) -> int:
        return len(self.pool)

    @property
    def evidence_text(self) -> str:
        """Raw concatenated chunk text -- for substring/number checking, not for
        showing a model (that is `format_evidence`)."""
        return " ".join(c["text"] for c in self.pool)


@dataclass(frozen=True)
class Failure:
    """Why an answer did not pass, in the shape the heal loop needs."""

    failure_type: str
    feedback: str
    claims: tuple[str, ...] = ()


Check = Callable[[VerifyContext], Optional[Failure]]


@dataclass(frozen=True)
class RegisteredCheck:
    name: str
    run: Check
    #: Routes this check runs on; `None` means every route.
    routes: Optional[frozenset] = field(default=_ALL_ROUTES)
    uses_llm: bool = False

    def applies_to(self, route: str) -> bool:
        return self.routes is None or route in self.routes


# ------------------------------------------------- deterministic checks (free)

# Cap on how many offending markers a failure message enumerates; the message is
# reinjected into the regenerate prompt, so it must stay short.
_MAX_REPORTED_MARKERS = 8


def _check_citations(ctx: VerifyContext) -> Optional[Failure]:
    """Every claim must carry a marker that resolves to real evidence."""
    markers = parse_markers(ctx.answer, ctx.n_evidence)
    if markers.malformed:
        return Failure("citations", "the answer contains malformed citation markers like [n]")
    if not markers.any_found:
        return Failure("citations", "the answer contains no [n] citations")
    if markers.out_of_range:
        # This feedback is fed back to the model verbatim on the regenerate pass,
        # so it is truncated: a wide range like "[1-500]" against 12 chunks parses
        # cleanly and would otherwise list hundreds of indices in the prompt.
        shown = list(markers.out_of_range[:_MAX_REPORTED_MARKERS])
        suffix = (
            f" (+{len(markers.out_of_range) - _MAX_REPORTED_MARKERS} more)"
            if len(markers.out_of_range) > _MAX_REPORTED_MARKERS
            else ""
        )
        return Failure(
            "citations",
            f"citation markers {shown}{suffix} do not match any "
            f"evidence (1..{ctx.n_evidence})",
        )
    return None


_NUM_RE = re.compile(
    r"\d+/\d+"                                   # fractions: 1/4, 1/135
    r"|\d+(?:,\d{3})*(?:\.\d+)?"                 # 4,000  93.3  175
    r"(?:\s*(?:%|percent|billion|million|thousand|[bmk]\b))?",
    re.IGNORECASE,
)

_MAGNITUDES = {"b": "billion", "m": "million", "k": "thousand", "percent": "%"}


def _canon_number(tok: str) -> str:
    t = tok.lower().replace(",", "").strip()
    t = re.sub(r"\s+", "", t)
    m = re.match(r"^(\d+(?:\.\d+)?)(%|percent|billion|million|thousand|b|m|k)$", t)
    if m:
        value, suffix = m.group(1), m.group(2)
        suffix = _MAGNITUDES.get(suffix, suffix)
        return f"{value}{suffix}"
    return t


def _extract_numbers(text: str) -> list[str]:
    return [m.group(0) for m in _NUM_RE.finditer(text)]


def _check_numbers(ctx: VerifyContext) -> Optional[Failure]:
    """No figure may appear in the answer that is absent from the evidence."""
    body = strip_markers(ctx.answer)  # citation indices are not claims
    evidence_text = ctx.evidence_text
    allowed = {
        _canon_number(t)
        for t in _extract_numbers(evidence_text) + _extract_numbers(ctx.question)
    }
    evidence_canon = re.sub(r"[,\s]", "", evidence_text.lower())
    offending = []
    for tok in _extract_numbers(body):
        canon = _canon_number(tok)
        if canon in allowed:
            continue
        # Boundary-anchored fallback: the canonical value must appear in the
        # evidence as a whole number, not as a substring of a larger one
        # ("400" must NOT pass because "14000" exists). Reject digit/decimal
        # continuation on either side; a sentence-final period still matches.
        pattern = r"(?<![\d.])" + re.escape(canon) + r"(?!\.?\d)"
        if re.search(pattern, evidence_canon):
            continue
        offending.append(tok)
    if offending:
        return Failure(
            "fabrication",
            "the answer contains values not present in the evidence: " + ", ".join(offending),
        )
    return None


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _check_coverage(ctx: VerifyContext) -> Optional[Failure]:
    """Each decomposed sub-question must be addressed by some answer sentence,
    measured by embedding similarity against COVERAGE_SIM_THRESHOLD."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(ctx.answer) if len(s.strip()) > 20]
    if not ctx.sub_questions or not sentences:
        return None
    q_vecs = np.asarray(embed_texts(list(ctx.sub_questions)), dtype="float32")
    s_vecs = np.asarray(embed_texts(sentences), dtype="float32")

    def _norm(v):
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return v / n

    sims = _norm(q_vecs) @ _norm(s_vecs).T  # (n_subq, n_sentences)
    missed = [
        sq
        for sq, row in zip(ctx.sub_questions, sims)
        if float(row.max()) < config.COVERAGE_SIM_THRESHOLD
    ]
    if missed:
        return Failure("coverage", "the answer does not address: " + "; ".join(missed))
    return None


# ---------------------------------------------------- LLM checks (one call each)

_RESPONSIVENESS_SYSTEM = (
    "You judge whether an answer directly addresses a question. "
    "Reply with exactly one word: yes or no."
)


def _check_responsiveness(ctx: VerifyContext) -> Optional[Failure]:
    raw = chat(
        _RESPONSIVENESS_SYSTEM,
        f"Question: {ctx.question}\n\nAnswer:\n{ctx.answer}\n\n"
        "Does the answer directly address what the question asked?",
        max_tokens=8,
        label="verify-responsiveness",
    )
    # Fail-open on garbage: the support check still runs, and looping on an
    # unparseable verdict would burn the heal budget for nothing.
    if raw.strip().lower().startswith("no"):
        return Failure("responsiveness", "the answer does not directly address the question")
    return None


_SUPPORT_SYSTEM = """You verify whether an answer is fully supported by the numbered evidence \
provided. Check EVERY claim in the answer against the evidence.

Respond in exactly this format:

If every claim is supported:
VERDICT: clean

If any claim is NOT supported by the evidence:
VERDICT: unsupported
CLAIMS:
- <the unsupported claim, restated as a short, self-contained search question>
- <another unsupported claim, restated as a short, self-contained search question>

Restate each unsupported claim as a search-friendly question, not a verbatim quote — \
these will be used as follow-up search queries. Output ONLY the verdict block, no \
other text."""


def _parse_support(raw: str) -> tuple[bool, list[str]]:
    text = raw.strip()
    verdict_match = re.search(r"VERDICT:\s*(clean|unsupported)", text, re.IGNORECASE)
    if verdict_match is None:
        return True, []
    if verdict_match.group(1).lower() == "clean":
        return True, []
    claims_section = text[verdict_match.end():]
    claims = [
        line.strip().lstrip("-").strip()
        for line in claims_section.splitlines()
        if line.strip().startswith("-")
    ]
    claims = [c for c in claims if c]
    if not claims:
        return True, []
    return False, claims


def _check_support(ctx: VerifyContext) -> Optional[Failure]:
    raw = chat(
        _SUPPORT_SYSTEM,
        f"Question: {ctx.question}\n\nAnswer:\n{ctx.answer}\n\n"
        f"Evidence:\n{format_evidence(ctx.pool)}\n\nVerdict:",
        label="verify-support",
    )
    clean, claims = _parse_support(raw)
    if clean:
        return None
    return Failure(
        "support",
        "claims lacking evidence: " + "; ".join(claims),
        claims=tuple(claims),
    )


# ------------------------------------------------------------- the registry

CHECKS: tuple[RegisteredCheck, ...] = (
    RegisteredCheck("citations", _check_citations, _ALL_ROUTES, uses_llm=False),
    RegisteredCheck("fabrication", _check_numbers, _ALL_ROUTES, uses_llm=False),
    RegisteredCheck("coverage", _check_coverage, _HARD_ONLY, uses_llm=False),
    RegisteredCheck("responsiveness", _check_responsiveness, _HARD_ONLY, uses_llm=True),
    RegisteredCheck("support", _check_support, _HARD_ONLY, uses_llm=True),
)


# Where each failure sends the graph. Evidence is present but the prose is wrong
# -> regenerate from the same pool. Evidence is genuinely missing -> go find more.
HEAL_ACTIONS: dict[str, str] = {
    "citations": "regenerate",
    "fabrication": "regenerate",
    "coverage": "regenerate",
    "responsiveness": "regenerate",
    "support": "research",
}


def _validate_registry() -> None:
    """Guards the registry's two invariants at import time, so a bad edit fails
    loudly at startup rather than quietly at the wrong moment in a heal loop."""
    seen_llm = False
    for check in CHECKS:
        if check.uses_llm:
            seen_llm = True
        elif seen_llm:
            raise AssertionError(
                f"CHECKS is misordered: deterministic check {check.name!r} runs after "
                "an LLM check, so a free failure would cost an LLM call"
            )
    missing = {c.name for c in CHECKS} - set(HEAL_ACTIONS)
    if missing:
        raise AssertionError(
            f"checks with no heal action: {sorted(missing)} -- they would silently "
            "fall back to 'regenerate'"
        )


_validate_registry()


def run_checks(ctx: VerifyContext) -> Optional[Failure]:
    """First applicable failure wins; later checks are not run."""
    for check in CHECKS:
        if not check.applies_to(ctx.route):
            continue
        failure = check.run(ctx)
        if failure is not None:
            return failure
    return None


# ------------------------------------------------------------------ the node

def _passed(reason: str) -> RAGState:
    return {
        "critique_clean": True,
        "failure_type": "",
        "verify_feedback": "",
        "unsupported_claims": [],
        "heal_action": "none",
        "verification_warnings": [],
        "trace": [f"verify → {reason}"],
    }


def verify(state: RAGState) -> RAGState:
    answer = state.get("answer", "")
    pool = cap_pool(state.get("retrieved", []))
    if not pool or not answer:
        return _passed("clean (no evidence/answer to check)")

    ctx = VerifyContext(
        question=state["question"],
        answer=answer,
        route=state.get("route", ""),
        sub_questions=tuple(state.get("sub_questions") or []),
        pool=pool,
    )

    failure = run_checks(ctx)
    if failure is None:
        log.info("verdict=clean")
        return _passed("clean")

    # Healing dispatch, bounded by the iteration budget. `iterations` is written
    # only by generate, so it counts answer attempts exactly.
    iterations = state.get("iterations", 0)
    budget = (
        config.MEDIUM_ITERATION_CAP
        if ctx.route == config.ROUTE_MEDIUM
        else config.MAX_ITERATIONS
    )
    if iterations >= budget:
        log.info("verdict=%s, budget exhausted (%d/%d) -> honest exit",
                 failure.failure_type, iterations, budget)
        return {
            "critique_clean": False,
            "failure_type": failure.failure_type,
            "verify_feedback": failure.feedback,
            "unsupported_claims": list(failure.claims),
            "heal_action": "none",
            "verification_warnings": [f"{failure.failure_type}: {failure.feedback}"],
            "answer": answer + (
                "\n\nNote: this answer could not be fully verified against the "
                f"indexed documents ({failure.feedback})."
            ),
            "trace": [f"verify → {failure.failure_type}, budget exhausted, honest exit"],
        }

    heal_action = HEAL_ACTIONS.get(failure.failure_type, "regenerate")
    log.info("verdict=%s -> %s (iteration %d/%d)",
             failure.failure_type, heal_action, iterations, budget)
    return {
        "critique_clean": False,
        "failure_type": failure.failure_type,
        "verify_feedback": failure.feedback,
        "unsupported_claims": list(failure.claims),
        "heal_action": heal_action,
        "verification_warnings": [],
        "trace": [f"verify → {failure.failure_type} → {heal_action}"],
    }
