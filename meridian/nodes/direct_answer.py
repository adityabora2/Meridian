from __future__ import annotations

from meridian.nodes.llm import chat
from meridian.state import RAGState

_SYSTEM = """You are a concise, accurate assistant answering a general-knowledge question \
about AI / machine learning. Answer directly from your own knowledge in 1-4 sentences. \
Do not fabricate citations or references — this answer is intentionally not grounded in \
any retrieved document."""


def direct_answer(state: RAGState) -> RAGState:
    answer = chat(_SYSTEM, state["question"], label="direct_answer")
    return {"answer": answer, "citations": [], "trace": ["direct_answer (no retrieval)"]}
