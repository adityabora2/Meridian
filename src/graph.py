from __future__ import annotations

try:
    from src import config
    from src.state import RAGState
except ImportError:
    import config  # type: ignore
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
