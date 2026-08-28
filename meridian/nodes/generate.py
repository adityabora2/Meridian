from __future__ import annotations

from meridian.evidence import cap_pool, format_evidence, resolve_citations
from meridian.logging_config import get_logger
from meridian.nodes.llm import chat
from meridian.state import RAGState

log = get_logger("generate")


_SYSTEM = """You answer questions strictly from the numbered evidence provided. Rules:
- Use ONLY the evidence below; do not add facts that aren't supported by it.
- Cite every claim with inline markers like [1] or [2], referring to the evidence numbers.
- If the evidence does not contain the answer, say so plainly instead of guessing.
- Be concise and specific."""


def generate(state: RAGState) -> RAGState:
    """Writes the grounded answer, and owns the iteration counter.

    Every heal path -- `regenerate` directly, `research` via search -- ends here,
    so counting generations counts attempts. Keeping the increment in this one
    node is what lets verify's budget check simply read the number.
    """
    retrieved = cap_pool(state.get("retrieved", []))
    iterations = state.get("iterations", 0) + 1

    if not retrieved:
        log.info("no evidence retrieved -> cannot answer from documents")
        return {
            "answer": "I couldn't find relevant evidence in the indexed documents to answer this.",
            "citations": [],
            "iterations": iterations,
            "trace": ["generate → no evidence"],
        }

    user = f"Question: {state['question']}\n\nEvidence:\n{format_evidence(retrieved)}\n\n"
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

    citations = resolve_citations(answer, retrieved)

    log.info("%d evidence chunks -> answer with %d citation(s)", len(retrieved), len(citations))
    if log.isEnabledFor(10):
        for c in citations:
            log.debug("  cite [%s] %s p%s", c["marker"], c["document_name"], c["page_number"])
    return {
        "answer": answer,
        "citations": citations,
        "iterations": iterations,
        "trace": [f"generate → answer with {len(citations)} citation(s) (iteration {iterations})"],
    }
