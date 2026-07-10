from __future__ import annotations

import re

try:
    from src.logging_config import get_logger
    from src.nodes.llm import chat
    from src.state import RAGState
except ImportError:
    from logging_config import get_logger  # type: ignore
    from nodes.llm import chat  # type: ignore
    from state import RAGState  # type: ignore

log = get_logger("critique")


_SYSTEM = """You verify whether an answer is fully supported by the numbered evidence \
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


def _format_evidence(retrieved: list[dict]) -> str:
    lines = []
    for i, c in enumerate(retrieved, start=1):
        loc = f"{c['document_name']} p{c['page_number']}"
        if c.get("section_heading"):
            loc += f" · {c['section_heading']}"
        lines.append(f"[{i}] ({loc}) {c['text']}")
    return "\n\n".join(lines)


def _parse_critique(raw: str) -> tuple[bool, list[str]]:
    text = raw.strip()
    verdict_match = re.search(r"VERDICT:\s*(clean|unsupported)", text, re.IGNORECASE)
    if verdict_match is None:
        return True, []

    verdict = verdict_match.group(1).lower()
    if verdict == "clean":
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


def critique(state: RAGState) -> RAGState:
    question = state["question"]
    answer = state.get("answer", "")
    retrieved = state.get("retrieved", [])

    trace = list(state.get("trace", []))

    if not retrieved or not answer:
        trace.append("critique → clean (no evidence/answer to check)")
        return {"critique_clean": True, "unsupported_claims": [], "trace": trace}

    evidence = _format_evidence(retrieved)
    user = (
        f"Question: {question}\n\nAnswer:\n{answer}\n\nEvidence:\n{evidence}\n\n"
        "Verdict:"
    )
    raw = chat(_SYSTEM, user, label="critique")
    clean, claims = _parse_critique(raw)

    log.info("verdict=%s%s", "clean" if clean else "unsupported",
             "" if clean else f" ({len(claims)} unsupported claim(s))")
    if log.isEnabledFor(10) and not clean:
        for c in claims:
            log.debug("  unsupported: %s", c)
    trace.append(
        "critique → clean" if clean else f"critique → unsupported ({len(claims)} claim(s))"
    )
    return {"critique_clean": clean, "unsupported_claims": claims, "trace": trace}


if __name__ == "__main__":
    state = {
        "question": "What optimizer and learning rate schedule does the Transformer use?",
        "answer": (
            "The Transformer uses the Adam optimizer [1] with a warmup-then-decay "
            "learning rate schedule [1], and was trained on exactly 8 V100 GPUs for "
            "12 days [2]."
        ),
        "retrieved": [
            {
                "chunk_id": "c1",
                "document_name": "attention_is_all_you_need.pdf",
                "page_number": 7,
                "section_heading": "Training",
                "text": (
                    "We used the Adam optimizer with beta_1=0.9, beta_2=0.98, and "
                    "varied the learning rate over the course of training with a "
                    "warmup followed by decay proportional to the inverse square "
                    "root of the step number."
                ),
            }
        ],
        "trace": [],
    }
    result = critique(state)
    print(f"Clean: {result['critique_clean']}")
    print(f"Unsupported claims: {result['unsupported_claims']}")
    print(f"Trace: {result['trace']}")
