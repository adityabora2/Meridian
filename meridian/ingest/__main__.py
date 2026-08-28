"""CLI: `python -m meridian.ingest` builds the index; `--self-test` runs the
synthetic-PDF pipeline check end to end."""
from __future__ import annotations

import argparse
from pathlib import Path

import fitz  # PyMuPDF

from meridian import config
from meridian.ingest import using_index
from meridian.ingest.chunking import build_chunks
from meridian.ingest.embedding import count_tokens
from meridian.ingest.store import build_index, search


def _write_synthetic_pdf(path: Path) -> None:
    doc = fitz.open()
    pages = [
        (
            "1  Introduction\n\n"
            "Adaptive retrieval-augmented generation decides how to retrieve before "
            "retrieving. It classifies each query by complexity and routes easy queries "
            "to a no-retrieval path, medium queries to a single-hop search, and hard "
            "queries to a multi-hop pipeline with a self-critique loop."
        ),
        (
            "2  Self-Critique\n\n"
            "The self-critique node checks every claim in a draft answer against the "
            "retrieved evidence. If it finds unsupported claims, the graph loops back to "
            "the search node. A hard cap of three iterations guarantees termination even "
            "when evidence is missing from the corpus."
        ),
        (
            "3  Chain of Verification\n\n"
            "Chain-of-Verification generates verification questions for each claim, "
            "answers them against the sources, and revises the final answer so that every "
            "returned statement is grounded in the retrieved chunks."
        ),
    ]
    for body in pages:
        page = doc.new_page()
        rect = fitz.Rect(72, 72, page.rect.width - 72, page.rect.height - 72)
        page.insert_textbox(rect, body, fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    doc.close()


def self_test() -> None:
    import tempfile

    print("=== ingest self-test ===")
    with tempfile.TemporaryDirectory() as tmp:
        docs = Path(tmp) / "documents"
        pdf = docs / "synthetic.pdf"
        _write_synthetic_pdf(pdf)
        print(f"Wrote synthetic PDF -> {pdf}")

        chunks = build_chunks(pdf)
        assert chunks, "no chunks produced"
        assert all(
            c.chunk_id and c.document_name == "synthetic.pdf" and c.document_title
            for c in chunks
        )
        assert all(c.page_number >= 1 for c in chunks)
        headings = {c.section_heading for c in chunks}
        print(f"Parsed {len(chunks)} chunks; detected headings: {sorted(headings)}")
        max_tokens = max(count_tokens(c.text) for c in chunks)
        print(f"Max chunk token count: {max_tokens} (window = {config.CHUNK_SIZE_TOKENS})")
        assert max_tokens <= config.CHUNK_SIZE_TOKENS + 2, "chunk exceeds token window"

        with using_index(Path(tmp) / "index"):
            n = build_index(docs)
            assert n == len(chunks)
            assert config.FAISS_INDEX_PATH.exists() and config.METADATA_PATH.exists()

            # No cache_clear needed here: using_index flushed on entry and
            # build_index does not load the index, so this reads the fresh one.
            results = search("What stops the self-critique loop from running forever?", k=3)
            assert results, "search returned nothing"
            top = results[0]
            print(f"Top hit (score={top['score']:.3f}) "
                  f"[{top['document_name']} p{top['page_number']} "
                  f"'{top['section_heading']}']: {top['text'][:90]}...")
            assert "iteration" in top["text"].lower() or "cap" in top["text"].lower() \
                or "three" in top["text"].lower(), "top hit is not the expected section"

    print("=== self-test PASSED ===")


def main() -> None:
    from meridian.logging_config import setup_logging

    setup_logging()

    parser = argparse.ArgumentParser(
        description="Build the FAISS index from data/documents/."
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the synthetic-PDF pipeline test instead of indexing real documents.",
    )
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        build_index()


if __name__ == "__main__":
    main()
