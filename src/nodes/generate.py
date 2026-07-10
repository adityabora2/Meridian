from __future__ import annotations

import re

try:
    from src.logging_config import get_logger
    from src.nodes.llm import chat
    from src.nodes.search import cap_pool
    from src.state import RAGState
except ImportError:
    from logging_config import get_logger  # type: ignore
    from nodes.llm import chat  # type: ignore
    from nodes.search import cap_pool  # type: ignore
    from state import RAGState  # type: ignore

log = get_logger("generate")


_SYSTEM = """You answer questions strictly from the numbered evidence provided. Rules:
- Use ONLY the evidence below; do not add facts that aren't supported by it.
- Cite every claim with inline markers like [1] or [2], referring to the evidence numbers.
- If the evidence does not contain the answer, say so plainly instead of guessing.
- Be concise and specific."""


def _format_evidence(retrieved: list[dict]) -> str:
    lines = []
    for i, c in enumerate(retrieved, start=1):
        loc = f"{c['document_name']} p{c['page_number']}"
        if c.get("section_heading"):
            loc += f" · {c['section_heading']}"
        lines.append(f"[{i}] ({loc}) {c['text']}")
    return "\n\n".join(lines)


def _cited_indices(answer: str, n: int) -> list[int]:
    found = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
    return sorted(i for i in found if 1 <= i <= n)


def generate(state: RAGState) -> RAGState:
    retrieved = cap_pool(state.get("retrieved", []))
    trace = list(state.get("trace", []))

    if not retrieved:
        log.info("no evidence retrieved -> cannot answer from documents")
        trace.append("generate → no evidence")
        return {
            "answer": "I couldn't find relevant evidence in the indexed documents to answer this.",
            "citations": [],
            "trace": trace,
        }

    evidence = _format_evidence(retrieved)
    user = f"Question: {state['question']}\n\nEvidence:\n{evidence}\n\n"
    feedback = state.get("verify_feedback", "")
    if feedback:
        user += (
            "PREVIOUS ATTEMPT FAILED VERIFICATION: "
            f"{feedback}. Rewrite the answer correcting this. Answer the "
            "question directly, use only the evidence, and cite with [n] "
            "markers that match the evidence numbers.\n\n"
        )
    user += "Answer with inline [n] citations:"
    answer = chat(_SYSTEM, user, label="generate")

    cited = _cited_indices(answer, len(retrieved))
    citations = []
    for i in cited:
        c = retrieved[i - 1]
        citations.append(
            {
                "marker": i,
                "chunk_id": c["chunk_id"],
                "document_name": c["document_name"],
                "page_number": c["page_number"],
                "section_heading": c.get("section_heading", ""),
                "score": c.get("score"),
            }
        )

    log.info("%d evidence chunks -> answer with %d citation(s)", len(retrieved), len(citations))
    if log.isEnabledFor(10):
        for c in citations:
            log.debug("  cite [%s] %s p%s", c["marker"], c["document_name"], c["page_number"])
    trace.append(f"generate → answer with {len(citations)} citation(s)")
    return {"answer": answer, "citations": citations, "trace": trace}
