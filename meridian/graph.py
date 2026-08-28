"""LangGraph assembly: the router branch, the mode paths, and the heal loop.

              ┌─ easy ──→ direct_answer ──────────────────────────────→ END
              │
    START →  router ─ meta ──→ corpus_info ─────────────────────────→ END
              │
              ├─ hard ──→ decompose ─┐
              │                      ↓
              └─ medium ─────────→ search ──→ generate ──→ verify ──→ END
                                     ↑           ↑            │  (none)
                                     │           └────────────┤  (regenerate)
                                     └── prepare_research ←───┘  (research)

The loop is bounded by `verify`'s budget check against `iterations`, which
`generate` alone increments -- so every cycle through it costs exactly one.
"""
from __future__ import annotations

import re

from meridian import config
from meridian.evidence import strip_markers
from meridian.nodes.corpus_info import corpus_info
from meridian.nodes.decompose import decompose
from meridian.nodes.direct_answer import direct_answer
from meridian.nodes.generate import generate
from meridian.nodes.router import route_question
from meridian.nodes.search import search_node
from meridian.nodes.verify import verify
from meridian.state import RAGState

# Where the router's label sends the graph. A table rather than an if-ladder, so
# adding a mode means adding a route constant and one entry here.
_ROUTE_DESTINATIONS: dict[str, str] = {
    config.ROUTE_EASY: "direct_answer",
    config.ROUTE_MEDIUM: "search",
    config.ROUTE_META: "corpus_info",
    config.ROUTE_HARD: "decompose",
}

# An unrecognized label must still terminate; decompose is the safe fallback
# (slower, never wrong) for the same reason the router keeps unscoped questions
# on the hard path.
_DEFAULT_DESTINATION = "decompose"


def route_from_router(state: RAGState) -> str:
    return _ROUTE_DESTINATIONS.get(state["route"], _DEFAULT_DESTINATION)


def route_from_verify(state: RAGState) -> str:
    return state.get("heal_action", "none")


def prepare_research(state: RAGState) -> RAGState:
    """Turns verify's unsupported claims into sanitized retrieval queries.
    sub_questions is never touched: the coverage check keeps verifying the
    ORIGINAL decomposition, not the retry queries."""
    claims = state.get("unsupported_claims", [])
    # Markers are replaced by a space (so "word[1]word" cannot fuse), then the
    # resulting whitespace runs are collapsed -- these strings become FAISS
    # queries, and ragged spacing is noise in the embedding.
    queries = [re.sub(r"\s+", " ", strip_markers(c)).strip() for c in claims]
    queries = [q for q in queries if q]
    return {"retry_queries": queries or [state["question"]]}


def build_graph():
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(RAGState)

    graph.add_node("router", route_question)
    graph.add_node("direct_answer", direct_answer)
    graph.add_node("decompose", decompose)
    graph.add_node("search", search_node)
    graph.add_node("generate", generate)
    graph.add_node("verify", verify)
    graph.add_node("prepare_research", prepare_research)
    graph.add_node("corpus_info", corpus_info)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_from_router,
        {dest: dest for dest in set(_ROUTE_DESTINATIONS.values())},
    )
    graph.add_edge("direct_answer", END)
    graph.add_edge("corpus_info", END)
    graph.add_edge("decompose", "search")
    graph.add_edge("search", "generate")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges(
        "verify",
        route_from_verify,
        {"none": END, "regenerate": "generate", "research": "prepare_research"},
    )
    graph.add_edge("prepare_research", "search")

    return graph.compile()
