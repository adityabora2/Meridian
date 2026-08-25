"""FAISS index construction, loading, and query-time retrieval."""
from __future__ import annotations

import json
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Optional

from meridian import config
from meridian.ingest.chunking import Chunk, build_chunks
from meridian.ingest.embedding import embed_texts


def build_index(docs_dir: Optional[Path] = None) -> int:
    import faiss

    docs_dir = docs_dir or config.DOCS_DIR
    pdfs = sorted(Path(docs_dir).glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"No PDFs found in {docs_dir}. Drop the documents there and re-run ingestion."
        )

    all_chunks: list[Chunk] = []
    for pdf in pdfs:
        chunks = build_chunks(pdf)
        print(f"  {pdf.name}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError("PDFs parsed but produced zero chunks (empty/scanned PDFs?).")

    print(f"Embedding {len(all_chunks)} chunks with {config.EMBEDDING_MODEL} ...")
    embeddings = embed_texts([c.text for c in all_chunks])

    index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
    index.add(embeddings)

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.FAISS_INDEX_PATH))
    with open(config.METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in all_chunks], f, ensure_ascii=False, indent=2)

    print(
        f"Indexed {len(all_chunks)} chunks from {len(pdfs)} PDF(s).\n"
        f"  index    -> {config.FAISS_INDEX_PATH}\n"
        f"  metadata -> {config.METADATA_PATH}"
    )
    return len(all_chunks)


@lru_cache(maxsize=1)
def load_index():
    import faiss

    if not config.FAISS_INDEX_PATH.exists() or not config.METADATA_PATH.exists():
        raise FileNotFoundError(
            "FAISS index not found. Run `python -m meridian.ingest` first to build it."
        )
    index = faiss.read_index(str(config.FAISS_INDEX_PATH))
    with open(config.METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = [Chunk(**d) for d in json.load(f)]
    return index, metadata


def search(
    query: str, k: Optional[int] = None, document_hint: Optional[str] = None
) -> list[dict]:
    k = k or config.TOP_K
    index, metadata = load_index()
    q = embed_texts([query])

    if document_hint is None:
        fetch_k = min(k, index.ntotal)
    else:
        # Over-fetch, then filter: a scoped query still searches the whole index,
        # so it needs headroom to find k hits inside the one target document.
        fetch_k = min(max(k * 4, k), index.ntotal)
    scores, ids = index.search(q, fetch_k)

    results: list[dict] = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        chunk = metadata[idx]
        if document_hint is not None and chunk.document_name != document_hint:
            continue
        row = asdict(chunk)
        row["score"] = float(score)
        results.append(row)
        if len(results) >= k:
            break
    return results


@lru_cache(maxsize=64)
def _page_one_rows(document_name: str) -> tuple[dict, ...]:
    """Memoized page-1 lookup. Cached because a scoped query pays a full walk of
    the metadata list otherwise, once per sub-question, on every iteration."""
    _, metadata = load_index()
    return tuple(
        asdict(chunk)
        for chunk in metadata
        if chunk.document_name == document_name and chunk.page_number == 1
    )


def page_one_chunks(document_name: str) -> list[dict]:
    """Returns the given document's page-1 chunks (title/author/abstract block)
    as scored result dicts. Used to boost metadata into the candidate set for
    queries scoped to a specific document, where a raw author/title block ranks
    too low semantically to surface on its own.

    Rows are copied out of the cache so a caller pooling them can never mutate
    the memoized originals."""
    # score 0.0 is neutral; _merge keeps these, and real hits still rank above.
    return [{**row, "score": 0.0} for row in _page_one_rows(document_name)]
