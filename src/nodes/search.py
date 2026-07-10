from __future__ import annotations

try:
    from src import config
    from src.ingest import match_document, page_one_chunks, search as faiss_search
    from src.logging_config import get_logger
    from src.state import RAGState
except ImportError:
    import config  # type: ignore
    from ingest import match_document, page_one_chunks, search as faiss_search  # type: ignore
    from logging_config import get_logger  # type: ignore
    from state import RAGState  # type: ignore

log = get_logger("search")


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
    boosted_docs: set[str] = set()
    n_boosted = 0
    for q in queries:
        document_hint = match_document(q)
        hits = faiss_search(q, k=config.TOP_K, document_hint=document_hint)
        fresh.extend(hits)
        boosted = 0
        if document_hint and document_hint not in boosted_docs:
            p1 = page_one_chunks(document_hint)
            fresh.extend(p1)
            boosted = len(p1)
            n_boosted += boosted
            boosted_docs.add(document_hint)
        top = max((h.get("score", 0.0) for h in hits), default=0.0)
        scope = f"scoped->{document_hint}" if document_hint else "whole corpus"
        log.info(
            "query %r | %s | %d hits, top=%.3f, +%d page-1",
            q[:60], scope, len(hits), top, boosted,
        )
        if log.isEnabledFor(10):  # DEBUG: per-chunk detail
            for h in hits:
                log.debug(
                    "  hit %s p%s score=%.3f",
                    h.get("document_name"), h.get("page_number"), h.get("score", 0.0),
                )

    pooled = _merge(state.get("retrieved", []), fresh)

    iteration = state.get("iterations", 0) + 1
    log.info(
        "search x%d -> %d hits (+%d page-1), %d pooled (iteration %d)",
        len(queries), len(fresh), n_boosted, len(pooled), iteration,
    )

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
