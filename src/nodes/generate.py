"""Generate node — answer from retrieved chunks WITH citations. (Mode 2 and Mode 3.)

Retrieved chunks are numbered [1], [2], ... and fed to the model, which must ground its
answer in them and cite with inline [n] markers. We then resolve which chunks were
actually cited and hand that subset to the UI as structured citations (document + page).

If nothing was retrieved (shouldn't happen on these paths, but defensively), we say so
rather than inventing an answer.
"""

from __future__ import annotations

import re

try:
    from src.nodes.llm import chat
    from src.state import RAGState
except ImportError:  # running from inside src/
    from nodes.llm import chat  # type: ignore
    from state import RAGState  # type: ignore


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
    """Return the 1-based evidence numbers actually referenced in the answer text."""
    found = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
    return sorted(i for i in found if 1 <= i <= n)


def generate(state: RAGState) -> RAGState:
    """Generate a cited answer from the pooled retrieved chunks."""
    retrieved = state.get("retrieved", [])
    trace = list(state.get("trace", []))

    if not retrieved:
        trace.append("generate → no evidence")
        return {
            "answer": "I couldn't find relevant evidence in the indexed papers to answer this.",
            "citations": [],
            "trace": trace,
        }

    evidence = _format_evidence(retrieved)
    user = f"Question: {state['question']}\n\nEvidence:\n{evidence}\n\nAnswer with inline [n] citations:"
    answer = chat(_SYSTEM, user)

    # Resolve which evidence items were cited so the UI can show exact sources.
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

    trace.append(f"generate → answer with {len(citations)} citation(s)")
    return {"answer": answer, "citations": citations, "trace": trace}
