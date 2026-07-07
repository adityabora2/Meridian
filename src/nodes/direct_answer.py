from __future__ import annotations

try:
    from src.nodes.llm import chat
    from src.state import RAGState
except ImportError:
    from nodes.llm import chat  # type: ignore
    from state import RAGState  # type: ignore


_SYSTEM = """You are a concise, accurate assistant answering a general-knowledge question \
about AI / machine learning. Answer directly from your own knowledge in 1-4 sentences. \
Do not fabricate citations or references — this answer is intentionally not grounded in \
any retrieved document."""


def direct_answer(state: RAGState) -> RAGState:
    answer = chat(_SYSTEM, state["question"])
    trace = list(state.get("trace", []))
    trace.append("direct_answer (no retrieval)")
    return {"answer": answer, "citations": [], "trace": trace}
