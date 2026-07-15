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

log = get_logger("decompose")


_SYSTEM = """You break down a hard, multi-hop question about a collection of indexed \
documents into 2-3 focused sub-questions that can each be answered by a SINGLE, \
independent document search.

Rules:
- Produce 2 or 3 sub-questions, never 1 and never more than 3.
- Each sub-question must be self-contained and independently searchable.
- Output ONLY the sub-questions, one per line.
- No numbering, no bullets, no explanation, no preamble."""

_MAX_SUB_QUESTIONS = 3

# Strips a leading numbering/bullet marker (e.g. "1. ", "2) ", "- ", "* ") in case
# the LLM ignores the "no numbering, no bullets" instruction — keeps decompose's
# fallback handling symmetric with verify.py's claim-bullet stripping, since both
# outputs land as raw FAISS query text in search_node.
_LEADING_MARKER_RE = re.compile(r"^(?:\d+[.)]|[-*])\s*")


def _parse_sub_questions(raw: str, *, fallback: str) -> list[str]:
    lines = [_LEADING_MARKER_RE.sub("", line.strip()) for line in raw.strip().splitlines()]
    questions = [line for line in lines if line]
    if not questions:
        return [fallback]
    return questions[:_MAX_SUB_QUESTIONS]


def decompose(state: RAGState) -> RAGState:
    question = state["question"]
    raw = chat(_SYSTEM, question, label="decompose")
    sub_questions = _parse_sub_questions(raw, fallback=question)

    log.info("decomposed into %d sub-question(s)", len(sub_questions))
    if log.isEnabledFor(10):
        for i, sq in enumerate(sub_questions, 1):
            log.debug("  sub-q %d: %s", i, sq)
    trace = list(state.get("trace", []))
    trace.append(f"decompose → {len(sub_questions)} sub-question(s)")
    return {"sub_questions": sub_questions, "trace": trace}


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else (
        "Compare how BERT and the Transformer paper each handle positional "
        "information, and explain any tradeoffs."
    )
    result = decompose({"question": q, "trace": []})
    print(f"Question: {q}\n")
    for i, sq in enumerate(result["sub_questions"], start=1):
        print(f"  {i}. {sq}")
    print(f"\nTrace: {result['trace']}")
