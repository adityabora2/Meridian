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

Classify the user's question into exactly one complexity level:

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

Respond with ONLY one word: easy, medium, or hard. No punctuation, no explanation."""


# Meta-question detection is done deterministically here, NOT by the LLM router:
# a small model over-triggers a "meta" category, misclassifying ordinary content
# questions (e.g. "who are the authors in xlnet") as corpus questions. A tight
# pattern is more reliable. A meta question is one asking WHAT the collection
# contains -- it pairs a collection noun (documents/files/papers/corpus) with a
# meta verb (loaded/indexed/available/...) or a "list/what ... are there" frame,
# and does NOT name a specific indexed document (that signals a content question).
_META_NOUN = r"(?:documents?|files?|papers?|pdfs?|sources?|corpus)"
_META_VERB = r"(?:load(?:ed)?|ingest(?:ed)?|index(?:ed)?|available|have|list|there|do you have|ask about)"
_META_PATTERNS = [
    # "what/which/how many documents are loaded/indexed/available/there"
    re.compile(rf"\b(?:what|which|how many|list(?: all)?)\b.*\b{_META_NOUN}\b.*\b{_META_VERB}\b"),
    # "what documents ..." collection-listing frames without an explicit verb
    re.compile(rf"\bwhat\b.*\ball\b.*\b{_META_NOUN}\b"),
    re.compile(rf"\b(?:list|show)\b.*\b{_META_NOUN}\b"),
    # "explain/describe the documents/corpus" (asking about the set, not content)
    re.compile(rf"\b(?:explain|describe|summari[sz]e)\b\s+(?:the\s+)?{_META_NOUN}\b"),
]


def is_meta_question(question: str) -> bool:
    """True if the question asks WHAT the indexed collection contains, rather
    than about the content of a document. A question that names a specific
    indexed document is treated as content (returns False), never meta."""
    text = question.strip().lower()
    if not any(p.search(text) for p in _META_PATTERNS):
        return False
    # A named document means the user is asking about that document's content,
    # not about the corpus listing -- e.g. "explain the bert document".
    if match_document(question):
        return False
    return True


def _parse_label(raw: str) -> str | None:
    text = raw.strip().lower()
    if text in (config.ROUTE_EASY, config.ROUTE_MEDIUM, config.ROUTE_HARD):
        return text
    m = re.search(r"\b(easy|medium|hard)\b", text)
    return m.group(1) if m else None


def route_question(state: RAGState) -> RAGState:
    question = state["question"]
    trace = list(state.get("trace", []))

    # Deterministic meta detection runs BEFORE the LLM router -- corpus-listing
    # questions are answered by corpus_info, not classified by the small model.
    if is_meta_question(question):
        trace.append("router → meta")
        return {
            "route": config.ROUTE_META,
            "mode_label": config.MODE_LABELS[config.ROUTE_META],
            "iterations": 0,
            "trace": trace,
        }

    raw = chat(_SYSTEM, question, max_tokens=8)
    label = _parse_label(raw)

    if label is None:
        label = config.ROUTE_MEDIUM

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
