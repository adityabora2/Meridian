from __future__ import annotations

import re

import numpy as np

try:
    from src import config
    from src.ingest import embed_texts
    from src.logging_config import get_logger
    from src.nodes.llm import chat
    from src.nodes.search import cap_pool
    from src.state import RAGState
except ImportError:
    import config  # type: ignore
    from ingest import embed_texts  # type: ignore
    from logging_config import get_logger  # type: ignore
    from nodes.llm import chat  # type: ignore
    from nodes.search import cap_pool  # type: ignore
    from state import RAGState  # type: ignore

log = get_logger("verify")

# Failure types dispatched to regeneration (evidence is present; the prose is
# wrong) versus re-search (evidence is genuinely missing).
_REGEN_FAILURES = {"citations", "fabrication", "coverage", "responsiveness"}

_MARKER_RE = re.compile(r"\[n?\d*\]")
_VALID_MARKER_RE = re.compile(r"\[(\d+)\]")
_MALFORMED_MARKER_RE = re.compile(r"\[n\d*\]")

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


def _check_citations(answer: str, n_evidence: int) -> tuple[bool, str]:
    if _MALFORMED_MARKER_RE.search(answer) or re.search(r"\[n\]", answer):
        return False, "the answer contains malformed citation markers like [n]"
    markers = [int(m) for m in _VALID_MARKER_RE.findall(answer)]
    if not markers:
        return False, "the answer contains no [n] citations"
    bad = sorted({m for m in markers if not (1 <= m <= n_evidence)})
    if bad:
        return False, f"citation markers {bad} do not match any evidence (1..{n_evidence})"
    return True, ""


def _check_numbers(answer: str, evidence_text: str, question: str) -> tuple[bool, list[str]]:
    body = _MARKER_RE.sub(" ", answer)  # citation indices are not claims
    allowed = {
        _canon_number(t)
        for t in _extract_numbers(evidence_text) + _extract_numbers(question)
    }
    evidence_canon = re.sub(r"[,\s]", "", evidence_text.lower())
    offending = []
    for tok in _extract_numbers(body):
        canon = _canon_number(tok)
        if canon in allowed or canon in evidence_canon:
            continue
        offending.append(tok)
    return (not offending), offending


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _check_coverage(sub_questions: list[str], answer: str) -> tuple[bool, list[str]]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(answer) if len(s.strip()) > 20]
    if not sub_questions or not sentences:
        return True, []
    q_vecs = np.asarray(embed_texts(list(sub_questions)), dtype="float32")
    s_vecs = np.asarray(embed_texts(sentences), dtype="float32")

    def _norm(v):
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return v / n

    sims = _norm(q_vecs) @ _norm(s_vecs).T  # (n_subq, n_sentences)
    missed = [
        sq
        for sq, row in zip(sub_questions, sims)
        if float(row.max()) < config.COVERAGE_SIM_THRESHOLD
    ]
    return (not missed), missed


_RESPONSIVENESS_SYSTEM = (
    "You judge whether an answer directly addresses a question. "
    "Reply with exactly one word: yes or no."
)


def _check_responsiveness(question: str, answer: str) -> bool:
    raw = chat(
        _RESPONSIVENESS_SYSTEM,
        f"Question: {question}\n\nAnswer:\n{answer}\n\n"
        "Does the answer directly address what the question asked?",
        max_tokens=8,
        label="verify-responsiveness",
    )
    # Fail-open on garbage: the support check still runs, and looping on an
    # unparseable verdict would burn the heal budget for nothing.
    return not raw.strip().lower().startswith("no")


# Support check: absorbed from the former critique node, unchanged behavior.
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


def _format_evidence(retrieved: list[dict]) -> str:
    lines = []
    for i, c in enumerate(retrieved, start=1):
        loc = f"{c['document_name']} p{c['page_number']}"
        if c.get("section_heading"):
            loc += f" · {c['section_heading']}"
        lines.append(f"[{i}] ({loc}) {c['text']}")
    return "\n\n".join(lines)


def _check_support(question: str, answer: str, evidence: str) -> tuple[bool, list[str]]:
    raw = chat(
        _SUPPORT_SYSTEM,
        f"Question: {question}\n\nAnswer:\n{answer}\n\nEvidence:\n{evidence}\n\nVerdict:",
        label="verify-support",
    )
    return _parse_support(raw)


def verify(state: RAGState) -> RAGState:
    question = state["question"]
    answer = state.get("answer", "")
    route = state.get("route", "")
    sub_questions = state.get("sub_questions") or []
    iterations = state.get("iterations", 0)
    trace = list(state.get("trace", []))

    pool = cap_pool(state.get("retrieved", []))
    if not pool or not answer:
        trace.append("verify → clean (no evidence/answer to check)")
        return {
            "critique_clean": True, "failure_type": "", "verify_feedback": "",
            "unsupported_claims": [], "heal_action": "none",
            "verification_warnings": [], "iterations": iterations,
            "answer": answer, "trace": trace,
        }

    evidence_text = " ".join(c["text"] for c in pool)
    failure_type = ""
    feedback = ""
    unsupported: list[str] = []

    # ---- Stage 1: deterministic, zero LLM calls ----
    ok, fb = _check_citations(answer, len(pool))
    if not ok:
        failure_type, feedback = "citations", fb
    if not failure_type:
        ok, offending = _check_numbers(answer, evidence_text, question)
        if not ok:
            failure_type = "fabrication"
            feedback = (
                "the answer contains values not present in the evidence: "
                + ", ".join(offending)
            )
    if not failure_type and route == config.ROUTE_HARD:
        ok, missed = _check_coverage(sub_questions, answer)
        if not ok:
            failure_type = "coverage"
            feedback = "the answer does not address: " + "; ".join(missed)

    # ---- Stage 2: binary LLM checks (hard route only, only if Stage 1 passed) ----
    if not failure_type and route == config.ROUTE_HARD:
        if not _check_responsiveness(question, answer):
            failure_type = "responsiveness"
            feedback = "the answer does not directly address the question"
        else:
            clean, claims = _check_support(question, answer, _format_evidence(pool))
            if not clean:
                failure_type = "support"
                unsupported = claims
                feedback = "claims lacking evidence: " + "; ".join(claims)

    if not failure_type:
        log.info("verdict=clean")
        trace.append("verify → clean")
        return {
            "critique_clean": True, "failure_type": "", "verify_feedback": "",
            "unsupported_claims": [], "heal_action": "none",
            "verification_warnings": [], "iterations": iterations,
            "answer": answer, "trace": trace,
        }

    # ---- Healing dispatch, bounded by the iteration budget ----
    budget = (
        config.MEDIUM_ITERATION_CAP
        if route == config.ROUTE_MEDIUM
        else config.MAX_ITERATIONS
    )
    if iterations >= budget:
        warning = f"{failure_type}: {feedback}"
        note = (
            "\n\nNote: this answer could not be fully verified against the "
            f"indexed documents ({feedback})."
        )
        log.info("verdict=%s, budget exhausted (%d/%d) -> honest exit",
                 failure_type, iterations, budget)
        trace.append(f"verify → {failure_type}, budget exhausted, honest exit")
        return {
            "critique_clean": False, "failure_type": failure_type,
            "verify_feedback": feedback, "unsupported_claims": unsupported,
            "heal_action": "none", "verification_warnings": [warning],
            "iterations": iterations, "answer": answer + note, "trace": trace,
        }

    if failure_type in _REGEN_FAILURES:
        heal_action = "regenerate"
        iterations += 1  # regeneration bypasses search_node's counter
    else:
        heal_action = "research"  # search_node will increment

    log.info("verdict=%s -> %s (iteration %d)", failure_type, heal_action, iterations)
    trace.append(f"verify → {failure_type} → {heal_action}")
    return {
        "critique_clean": False, "failure_type": failure_type,
        "verify_feedback": feedback, "unsupported_claims": unsupported,
        "heal_action": heal_action, "verification_warnings": [],
        "iterations": iterations, "answer": answer, "trace": trace,
    }
