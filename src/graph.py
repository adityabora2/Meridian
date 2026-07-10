from __future__ import annotations

import re

try:
    from src import config
    from src.nodes.corpus_info import corpus_info
    from src.nodes.decompose import decompose
    from src.nodes.direct_answer import direct_answer
    from src.nodes.generate import generate
    from src.nodes.router import route_question
    from src.nodes.search import search_node
    from src.nodes.verify import verify
    from src.state import RAGState
except ImportError:
    import config  # type: ignore
    from nodes.corpus_info import corpus_info  # type: ignore
    from nodes.decompose import decompose  # type: ignore
    from nodes.direct_answer import direct_answer  # type: ignore
    from nodes.generate import generate  # type: ignore
    from nodes.router import route_question  # type: ignore
    from nodes.search import search_node  # type: ignore
    from nodes.verify import verify  # type: ignore
    from state import RAGState  # type: ignore


def route_from_router(state: RAGState) -> str:
    route = state["route"]
    if route == config.ROUTE_EASY:
        return "direct_answer"
    if route == config.ROUTE_MEDIUM:
        return "search"
    if route == config.ROUTE_META:
        return "corpus_info"
    return "decompose"


def route_from_verify(state: RAGState) -> str:
    return state.get("heal_action", "none")


def prepare_research(state: RAGState) -> RAGState:
    """Turns verify's unsupported claims into sanitized retrieval queries.
    sub_questions is never touched: the coverage check keeps verifying the
    ORIGINAL decomposition, not the retry queries."""
    claims = state.get("unsupported_claims", [])
    queries = [re.sub(r"\[n?\d*\]", "", c).strip() for c in claims]
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
        {
            "direct_answer": "direct_answer",
            "search": "search",
            "decompose": "decompose",
            "corpus_info": "corpus_info",
        },
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
