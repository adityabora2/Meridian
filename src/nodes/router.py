from __future__ import annotations

import re

try:
    from src import config
    from src.nodes.llm import chat
    from src.state import RAGState
    from src.ingest import match_document
except ImportError:
    import config  # type: ignore
    from nodes.llm import chat  # type: ignore
    from state import RAGState  # type: ignore
    from ingest import match_document  # type: ignore


_SYSTEM = """You are a query-complexity router for a document Q&A system over an \
indexed collection of documents.

Classify the user's question into exactly one category:

- easy: General knowledge or definitional questions a strong LLM can answer correctly \
WITHOUT looking at any document. No retrieval needed.
  Examples: "What does an acronym like RAG stand for?", "What is a vector database?"

- medium: The answer requires grounding in the documents, but a SINGLE focused search \
will surface it. One fact or concept, findable in one place.
  Examples: "What does document X say about topic Y?", \
"Which method does the source material describe for doing Z?"

- hard: The question is multi-hop or cross-document. Answering it needs several \
searches and evidence chained across sub-questions.
  Examples: "How does the approach in one document differ from another, and what do \
they share?", "Compare how several sources each handle the same underlying problem."

- meta: A question about the document collection or the system ITSELF, not about the \
content of any document. Answerable by listing what is indexed.
  Examples: "What documents are loaded?", "What files can I ask about?", \
"Which papers do you have?", "How many documents are indexed?"

Respond with ONLY one word: easy, medium, hard, or meta. No punctuation, no explanation."""


def _parse_label(raw: str) -> str | None:
    text = raw.strip().lower()
    if text in (config.ROUTE_EASY, config.ROUTE_MEDIUM, config.ROUTE_HARD, config.ROUTE_META):
        return text
    m = re.search(r"\b(easy|medium|hard|meta)\b", text)
    return m.group(1) if m else None


def route_question(state: RAGState) -> RAGState:
    question = state["question"]
    raw = chat(_SYSTEM, question, max_tokens=8)
    label = _parse_label(raw)

    if label is None:
        label = config.ROUTE_MEDIUM

    trace = list(state.get("trace", []))
    trace.append(f"router → {label}")

    # Document-aware upgrade: a general-knowledge (easy) question that names an
    # indexed document should be answered FROM that document, not model memory.
    if label == config.ROUTE_EASY:
        matched = match_document(question)
        if matched:
            label = config.ROUTE_MEDIUM
            trace.append(f"router → easy upgraded to medium (names {matched})")

    return {
        "route": label,
        "mode_label": config.MODE_LABELS[label],
        "iterations": 0,
        "trace": trace,
    }
