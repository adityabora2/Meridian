from __future__ import annotations

try:
    from src import config
    from src.nodes.critique import critique
    from src.nodes.decompose import decompose
    from src.nodes.direct_answer import direct_answer
    from src.nodes.generate import generate
    from src.nodes.router import route_question
    from src.nodes.search import search_node
    from src.state import RAGState
except ImportError:
    import config  # type: ignore
    from nodes.critique import critique  # type: ignore
    from nodes.decompose import decompose  # type: ignore
    from nodes.direct_answer import direct_answer  # type: ignore
    from nodes.generate import generate  # type: ignore
    from nodes.router import route_question  # type: ignore
    from nodes.search import search_node  # type: ignore
    from state import RAGState  # type: ignore


def route_from_router(state: RAGState) -> str:
    route = state["route"]
    if route == config.ROUTE_EASY:
        return "direct_answer"
    if route == config.ROUTE_MEDIUM:
        return "search"
    return "decompose"


def route_from_critique(state: RAGState) -> str:
    if state.get("critique_clean", True):
        return "end"
    if state.get("iterations", 0) < config.MAX_ITERATIONS:
        return "prepare_retry"
    return "end"


def prepare_retry(state: RAGState) -> RAGState:
    return {"sub_questions": state["unsupported_claims"]}


def build_graph():
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(RAGState)

    graph.add_node("router", route_question)
    graph.add_node("direct_answer", direct_answer)
    graph.add_node("decompose", decompose)
    graph.add_node("search", search_node)
    graph.add_node("generate", generate)
    graph.add_node("critique", critique)
    graph.add_node("prepare_retry", prepare_retry)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_from_router,
        {"direct_answer": "direct_answer", "search": "search", "decompose": "decompose"},
    )
    graph.add_edge("direct_answer", END)
    graph.add_edge("decompose", "search")
    graph.add_edge("search", "generate")
    graph.add_conditional_edges(
        "generate",
        lambda state: "critique" if state.get("sub_questions") else "end_mode2",
        {"critique": "critique", "end_mode2": END},
    )
    graph.add_conditional_edges(
        "critique",
        route_from_critique,
        {"prepare_retry": "prepare_retry", "end": END},
    )
    graph.add_edge("prepare_retry", "search")

    return graph.compile()
