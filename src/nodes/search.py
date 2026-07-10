from __future__ import annotations

try:
    from src import config
    from src.ingest import match_document, search as faiss_search
    from src.state import RAGState
except ImportError:
    import config  # type: ignore
    from ingest import match_document, search as faiss_search  # type: ignore
    from state import RAGState  # type: ignore


def _merge(existing: list[dict], new: list[dict]) -> list[dict]:
    by_id: dict[str, dict] = {c["chunk_id"]: c for c in existing}
    for c in new:
        prev = by_id.get(c["chunk_id"])
        if prev is None or c.get("score", 0.0) > prev.get("score", 0.0):
            by_id[c["chunk_id"]] = c
    return sorted(by_id.values(), key=lambda c: c.get("score", 0.0), reverse=True)


def search_node(state: RAGState) -> RAGState:
    sub_questions = state.get("sub_questions") or []
    queries = sub_questions if sub_questions else [state["question"]]

    fresh: list[dict] = []
    for q in queries:
        document_hint = match_document(q)
        fresh.extend(faiss_search(q, k=config.TOP_K, document_hint=document_hint))

    pooled = _merge(state.get("retrieved", []), fresh)

    trace = list(state.get("trace", []))
    n_q = len(queries)
    trace.append(
        f"search ×{n_q} → {len(fresh)} hits, {len(pooled)} pooled "
        f"(iteration {state.get('iterations', 0) + 1})"
    )
    return {
        "retrieved": pooled,
        "iterations": state.get("iterations", 0) + 1,
        "trace": trace,
    }
