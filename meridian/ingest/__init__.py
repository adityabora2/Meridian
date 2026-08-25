"""Ingestion and retrieval.

Split by concern -- `pdf` (layout heuristics), `chunking` (the Chunk record and
token windowing), `embedding` (the shared model), `store` (FAISS build/load/
search), `matching` (document-name resolution) -- and re-exported here so
callers keep importing one name: `from meridian.ingest import search`.
"""
from __future__ import annotations

from contextlib import contextmanager

from meridian import config
from meridian.ingest.chunking import Chunk, build_chunks, chunk_text
from meridian.ingest.embedding import count_tokens, embed_texts
from meridian.ingest.matching import (
    implicated_documents,
    match_document,
)
from meridian.ingest.pdf import extract_title, parse_pdf
from meridian.ingest.store import (
    build_index,
    load_index,
    page_one_chunks,
    search,
)

__all__ = [
    "Chunk",
    "build_chunks",
    "build_index",
    "chunk_text",
    "clear_caches",
    "count_tokens",
    "embed_texts",
    "extract_title",
    "implicated_documents",
    "load_index",
    "match_document",
    "page_one_chunks",
    "parse_pdf",
    "search",
    "using_index",
]


def clear_caches() -> None:
    """Drops every index-derived memo in one call.

    The loaded index, the page-1 lookup, and the document-match corpus are all
    cached off the same on-disk artifacts, so they must be invalidated
    together -- forgetting one leaves a stale corpus behind a fresh index.
    """
    from meridian.ingest import matching, store

    store.load_index.cache_clear()
    store._page_one_rows.cache_clear()
    matching._corpus_from_index.cache_clear()
    matching._tokenize.cache_clear()


@contextmanager
def using_index(index_dir):
    """Point ingestion and retrieval at another index directory for the duration
    of the block, with the caches flushed on the way in and on the way out.

    `config.index_at` only swaps the paths. Swapping paths without flushing
    leaves the previous index's chunks memoized behind the new paths -- so the
    two operations are fused here rather than left as a rule to remember.
    """
    with config.index_at(index_dir) as resolved:
        clear_caches()
        try:
            yield resolved
        finally:
            clear_caches()
