from __future__ import annotations

import re

try:
    from src import config
    from src.nodes.llm import chat
    from src.state import RAGState
except ImportError:
    import config  # type: ignore
    from nodes.llm import chat  # type: ignore
    from state import RAGState  # type: ignore


_SYSTEM = """You are a query-complexity router for a document Q&A system about \
AI research papers (Adaptive-RAG, Self-RAG, Chain-of-Verification, and related work).

Classify the user's question into exactly one complexity level:

- easy: General knowledge or definitional questions a strong LLM can answer correctly \
WITHOUT looking at any document. No retrieval needed.
  Examples: "What does RAG stand for?", "What is a vector database?"

- medium: The answer requires grounding in the papers, but a SINGLE focused search will \
surface it. One fact or concept, findable in one place.
  Examples: "What iteration cap does the self-critique loop use?", \
"Which dataset did Self-RAG evaluate on?"

- hard: The question is multi-hop or cross-document. Answering it needs several searches \
and evidence chained across sub-questions.
  Examples: "How does Adaptive-RAG's routing differ from Self-RAG's reflection, and \
what do they share?", "Compare the verification strategies across all three papers."

Respond with ONLY one word: easy, medium, or hard. No punctuation, no explanation."""


def _parse_label(raw: str) -> str | None:
    text = raw.strip().lower()
    if text in (config.ROUTE_EASY, config.ROUTE_MEDIUM, config.ROUTE_HARD):
        return text
    m = re.search(r"\b(easy|medium|hard)\b", text)
    return m.group(1) if m else None


def route_question(state: RAGState) -> RAGState:
    question = state["question"]
    raw = chat(_SYSTEM, question, max_tokens=8)
    label = _parse_label(raw)

    if label is None:
        label = config.ROUTE_MEDIUM

    trace = list(state.get("trace", []))
    trace.append(f"router → {label}")
    return {
        "route": label,
        "mode_label": config.MODE_LABELS[label],
        "iterations": 0,
        "trace": trace,
    }
