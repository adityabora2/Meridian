"""Corpus ingestion: PDF -> chunks -> embeddings -> persisted FAISS index.

Run standalone to (re)build the index from every PDF in ``data/papers/``:

    python -m src.ingest                # build/rebuild the index
    python -m src.ingest --self-test    # build a synthetic PDF and prove the pipeline

The graph's search node imports :func:`load_index` and :func:`search` from here, so
this module is both the offline builder and the runtime retrieval helper (there is no
separate ``retriever.py`` — see BUILD_LOG deviation D2).

Nothing here touches AWS or any network service except the one-time, cached download of
the local sentence-transformers model.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import numpy as np

try:  # allow both `python -m src.ingest` and `from src import ingest`
    from src import config
except ImportError:  # running from inside src/
    import config  # type: ignore


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Chunk:
    """One retrievable unit of text plus the metadata the spec requires."""

    chunk_id: str            # e.g. "adaptive-rag.pdf::p3::c2"
    text: str
    page_number: int         # 1-indexed
    section_heading: str     # best-effort; "" if none detected
    document_name: str       # source PDF filename


# --------------------------------------------------------------------------- #
# Lazy singletons — the model is heavy; load it once per process.
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _tokenizer():
    # Same tokenizer the embedding model uses, so "256 tokens" means exactly what the
    # embedder sees — chunk boundaries line up with the 256-token embedding window.
    return _model().tokenizer


def _count_tokens(text: str) -> int:
    return len(_tokenizer().encode(text, add_special_tokens=False))


# --------------------------------------------------------------------------- #
# PDF parsing + best-effort section heading detection
# --------------------------------------------------------------------------- #
# A heading looks like: short line, often numbered ("3", "3.1", "4.2.1"), or a small set
# of Title-Case words. This is heuristic by design — the spec says "best-effort".
_HEADING_NUMBERED = re.compile(r"^\s*(\d+(\.\d+)*)\s+[A-Z].{0,80}$")
_HEADING_KEYWORDS = re.compile(
    r"^\s*(abstract|introduction|related work|background|method(s|ology)?|"
    r"approach|experiments?|results?|discussion|conclusions?|references|"
    r"appendix|acknowledgm?ents?)\s*$",
    re.IGNORECASE,
)


def _looks_like_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 90:
        return False
    if _HEADING_KEYWORDS.match(line):
        return True
    if _HEADING_NUMBERED.match(line):
        return True
    return False


def _iter_page_blocks(page: "fitz.Page"):
    """Yield (text, is_heading_candidate) for text blocks on a page, in reading order.

    We use ``page.get_text("blocks")`` which returns layout blocks; the first line of a
    short block is our heading candidate. Font-size analysis would be more precise but
    "blocks" is robust and dependency-light, which fits a best-effort heading pass.
    """
    blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
    # Sort top-to-bottom, then left-to-right for rough reading order.
    blocks = sorted(blocks, key=lambda b: (round(b[1], 1), round(b[0], 1)))
    for b in blocks:
        text = (b[4] or "").strip()
        if text:
            yield text


def parse_pdf(pdf_path: Path) -> list[tuple[int, str, str]]:
    """Return a list of (page_number, section_heading, text) segments for one PDF.

    Each segment is a run of body text under the most recently seen heading. A new
    heading starts a new segment. This keeps ``section_heading`` attached to the right
    text without needing a full document outline.
    """
    segments: list[tuple[int, str, str]] = []
    current_heading = ""
    doc = fitz.open(pdf_path)
    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            page_number = page_index + 1
            buffer: list[str] = []
            for block_text in _iter_page_blocks(page):
                first_line = block_text.splitlines()[0].strip()
                if _looks_like_heading(first_line) and len(block_text) < 120:
                    # flush what we have under the previous heading
                    if buffer:
                        segments.append((page_number, current_heading, "\n".join(buffer)))
                        buffer = []
                    current_heading = first_line
                else:
                    buffer.append(block_text)
            if buffer:
                segments.append((page_number, current_heading, "\n".join(buffer)))
    finally:
        doc.close()
    return segments


# --------------------------------------------------------------------------- #
# Chunking — 256 tokens, 50 overlap (see config + BUILD_LOG D1)
# --------------------------------------------------------------------------- #
def _clean(text: str) -> str:
    # Collapse hyphenated line breaks ("infor-\nmation" -> "information") and whitespace.
    text = re.sub(r"-\n(?=\w)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str) -> list[str]:
    """Split text into ~256-token windows with 50-token overlap.

    Token counts come from the embedder's own tokenizer so the window matches exactly
    what the embedding model sees. We slice the ORIGINAL string using character offsets
    (``return_offsets_mapping``) rather than decoding token IDs — MiniLM's WordPiece
    tokenizer is lowercasing and lossy, so decoding would corrupt casing/punctuation in
    the stored chunk (bad for citations and for the LLM). Offsets keep the text verbatim.
    """
    text = _clean(text)
    if not text:
        return []
    tok = _tokenizer()
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]  # list of (char_start, char_end) per token
    n = len(offsets)
    if n == 0:
        return []
    size = config.CHUNK_SIZE_TOKENS
    overlap = config.CHUNK_OVERLAP_TOKENS
    step = max(1, size - overlap)
    chunks: list[str] = []
    for start in range(0, n, step):
        window = offsets[start : start + size]
        if not window:
            break
        char_start = window[0][0]
        char_end = window[-1][1]
        piece = text[char_start:char_end].strip()
        if piece:
            chunks.append(piece)
        if start + size >= n:
            break
    return chunks


def build_chunks(pdf_path: Path) -> list[Chunk]:
    """Parse one PDF and produce fully-tagged Chunk objects."""
    document_name = pdf_path.name
    chunks: list[Chunk] = []
    per_page_counter: dict[int, int] = {}
    for page_number, heading, text in parse_pdf(pdf_path):
        for piece in chunk_text(text):
            idx = per_page_counter.get(page_number, 0)
            per_page_counter[page_number] = idx + 1
            chunk_id = f"{document_name}::p{page_number}::c{idx}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=piece,
                    page_number=page_number,
                    section_heading=heading,
                    document_name=document_name,
                )
            )
    return chunks


# --------------------------------------------------------------------------- #
# Embedding + FAISS index build/persist
# --------------------------------------------------------------------------- #
def embed_texts(texts: list[str]) -> np.ndarray:
    """Return L2-normalized float32 embeddings so inner-product == cosine similarity."""
    vecs = _model().encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,  # -> cosine similarity via inner product
        show_progress_bar=False,
    )
    return np.asarray(vecs, dtype="float32")


def build_index(papers_dir: Optional[Path] = None) -> int:
    """Parse every PDF in ``papers_dir``, embed, build a FAISS index, and persist it.

    Returns the number of chunks indexed. Raises if there are no PDFs.
    """
    import faiss

    papers_dir = papers_dir or config.PAPERS_DIR
    pdfs = sorted(Path(papers_dir).glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"No PDFs found in {papers_dir}. Drop the papers there and re-run ingestion."
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

    # Inner-product index over normalized vectors == cosine similarity.
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


# --------------------------------------------------------------------------- #
# Runtime retrieval helpers (imported by the search node)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def load_index():
    """Load the persisted FAISS index + chunk metadata. Cached for the process lifetime."""
    import faiss

    if not config.FAISS_INDEX_PATH.exists() or not config.METADATA_PATH.exists():
        raise FileNotFoundError(
            "FAISS index not found. Run `python -m src.ingest` first to build it."
        )
    index = faiss.read_index(str(config.FAISS_INDEX_PATH))
    with open(config.METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = [Chunk(**d) for d in json.load(f)]
    return index, metadata


def search(query: str, k: Optional[int] = None) -> list[dict]:
    """Embed ``query`` and return the top-k chunks as dicts with a similarity ``score``.

    Each result: {chunk_id, text, page_number, section_heading, document_name, score}.
    This is the single retrieval entry point the graph's search node calls.
    """
    k = k or config.TOP_K
    index, metadata = load_index()
    q = embed_texts([query])
    scores, ids = index.search(q, min(k, index.ntotal))
    results: list[dict] = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:  # FAISS pads with -1 when fewer than k results exist
            continue
        chunk = metadata[idx]
        row = asdict(chunk)
        row["score"] = float(score)
        results.append(row)
    return results


# --------------------------------------------------------------------------- #
# Standalone self-test: build a synthetic PDF and prove the whole pipeline.
# --------------------------------------------------------------------------- #
def _write_synthetic_pdf(path: Path) -> None:
    """Create a tiny multi-section PDF so ingestion can be tested without real papers."""
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
        # insert_textbox wraps text inside a rectangle so long paragraphs stay on-page
        # and are fully extractable (insert_text at a point would run off the edge and
        # the overflow would be lost at parse time).
        rect = fitz.Rect(72, 72, page.rect.width - 72, page.rect.height - 72)
        page.insert_textbox(rect, body, fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    doc.close()


def self_test() -> None:
    """Build a synthetic PDF, run the full pipeline, and assert it works end-to-end."""
    import tempfile

    print("=== ingest self-test ===")
    with tempfile.TemporaryDirectory() as tmp:
        papers = Path(tmp) / "papers"
        pdf = papers / "synthetic.pdf"
        _write_synthetic_pdf(pdf)
        print(f"Wrote synthetic PDF -> {pdf}")

        # 1. parse + chunk + metadata
        chunks = build_chunks(pdf)
        assert chunks, "no chunks produced"
        assert all(c.chunk_id and c.document_name == "synthetic.pdf" for c in chunks)
        assert all(c.page_number >= 1 for c in chunks)
        headings = {c.section_heading for c in chunks}
        print(f"Parsed {len(chunks)} chunks; detected headings: {sorted(headings)}")
        max_tokens = max(_count_tokens(c.text) for c in chunks)
        print(f"Max chunk token count: {max_tokens} (window = {config.CHUNK_SIZE_TOKENS})")
        assert max_tokens <= config.CHUNK_SIZE_TOKENS + 2, "chunk exceeds token window"

        # 2. build + persist index (redirect config paths into the temp dir)
        orig_index, orig_meta, orig_dir = (
            config.FAISS_INDEX_PATH,
            config.METADATA_PATH,
            config.INDEX_DIR,
        )
        config.INDEX_DIR = Path(tmp) / "index"
        config.FAISS_INDEX_PATH = config.INDEX_DIR / "faiss.index"
        config.METADATA_PATH = config.INDEX_DIR / "metadata.json"
        load_index.cache_clear()
        try:
            n = build_index(papers)
            assert n == len(chunks)
            assert config.FAISS_INDEX_PATH.exists() and config.METADATA_PATH.exists()

            # 3. reload from disk + query
            load_index.cache_clear()
            results = search("What stops the self-critique loop from running forever?", k=3)
            assert results, "search returned nothing"
            top = results[0]
            print(f"Top hit (score={top['score']:.3f}) "
                  f"[{top['document_name']} p{top['page_number']} "
                  f"'{top['section_heading']}']: {top['text'][:90]}...")
            # The relevant answer lives in the "Self-Critique" section.
            assert "iteration" in top["text"].lower() or "cap" in top["text"].lower() \
                or "three" in top["text"].lower(), "top hit is not the expected section"
        finally:
            config.FAISS_INDEX_PATH, config.METADATA_PATH, config.INDEX_DIR = (
                orig_index, orig_meta, orig_dir,
            )
            load_index.cache_clear()

    print("=== self-test PASSED ===")


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Build the FAISS index from data/papers/.")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the synthetic-PDF pipeline test instead of indexing real papers.",
    )
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        build_index()


if __name__ == "__main__":
    main()
