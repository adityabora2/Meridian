from __future__ import annotations

from meridian import config
from meridian.ingest import match_document, page_one_chunks, search as faiss_search
from meridian.logging_config import get_logger
from meridian.state import RAGState

log = get_logger("search")


def search_node(state: RAGState) -> RAGState:
    """Runs one retrieval round and returns only what it found.

    Pooling across rounds is the `retrieved` channel's reducer
    (`state.merge_retrieved`), not this node's job -- the heal loop re-enters
    here with different queries and the evidence must accumulate.
    """
    sub_questions = state.get("sub_questions") or []
    retry_queries = state.get("retry_queries") or []
    # Re-search retries (from verify's support failures) take priority; they
    # never overwrite sub_questions, which the coverage check still needs.
    queries = retry_queries or sub_questions or [state["question"]]

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

    log.info("search x%d -> %d hits (+%d page-1)", len(queries), len(fresh), n_boosted)

    return {
        "retrieved": fresh,
        "trace": [f"search ×{len(queries)} → {len(fresh)} hits (+{n_boosted} page-1)"],
        "retry_queries": [],
    }
