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

try:
    from src import config
except ImportError:
    import config  # type: ignore


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page_number: int
    section_heading: str
    document_name: str


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _tokenizer():
    return _model().tokenizer


def _count_tokens(text: str) -> int:
    return len(_tokenizer().encode(text, add_special_tokens=False))


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
    blocks = page.get_text("blocks")
    blocks = sorted(blocks, key=lambda b: (round(b[1], 1), round(b[0], 1)))
    for b in blocks:
        text = (b[4] or "").strip()
        if text:
            yield text


def parse_pdf(pdf_path: Path) -> list[tuple[int, str, str]]:
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


def _clean(text: str) -> str:
    text = re.sub(r"-\n(?=\w)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str) -> list[str]:
    text = _clean(text)
    if not text:
        return []
    tok = _tokenizer()
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
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


def embed_texts(texts: list[str]) -> np.ndarray:
    vecs = _model().encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vecs, dtype="float32")


def build_index(papers_dir: Optional[Path] = None) -> int:
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
            "FAISS index not found. Run `python -m src.ingest` first to build it."
        )
    index = faiss.read_index(str(config.FAISS_INDEX_PATH))
    with open(config.METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = [Chunk(**d) for d in json.load(f)]
    return index, metadata


def search(query: str, k: Optional[int] = None) -> list[dict]:
    k = k or config.TOP_K
    index, metadata = load_index()
    q = embed_texts([query])
    scores, ids = index.search(q, min(k, index.ntotal))
    results: list[dict] = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        chunk = metadata[idx]
        row = asdict(chunk)
        row["score"] = float(score)
        results.append(row)
    return results


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
        papers = Path(tmp) / "papers"
        pdf = papers / "synthetic.pdf"
        _write_synthetic_pdf(pdf)
        print(f"Wrote synthetic PDF -> {pdf}")

        chunks = build_chunks(pdf)
        assert chunks, "no chunks produced"
        assert all(c.chunk_id and c.document_name == "synthetic.pdf" for c in chunks)
        assert all(c.page_number >= 1 for c in chunks)
        headings = {c.section_heading for c in chunks}
        print(f"Parsed {len(chunks)} chunks; detected headings: {sorted(headings)}")
        max_tokens = max(_count_tokens(c.text) for c in chunks)
        print(f"Max chunk token count: {max_tokens} (window = {config.CHUNK_SIZE_TOKENS})")
        assert max_tokens <= config.CHUNK_SIZE_TOKENS + 2, "chunk exceeds token window"

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

            load_index.cache_clear()
            results = search("What stops the self-critique loop from running forever?", k=3)
            assert results, "search returned nothing"
            top = results[0]
            print(f"Top hit (score={top['score']:.3f}) "
                  f"[{top['document_name']} p{top['page_number']} "
                  f"'{top['section_heading']}']: {top['text'][:90]}...")
            assert "iteration" in top["text"].lower() or "cap" in top["text"].lower() \
                or "three" in top["text"].lower(), "top hit is not the expected section"
        finally:
            config.FAISS_INDEX_PATH, config.METADATA_PATH, config.INDEX_DIR = (
                orig_index, orig_meta, orig_dir,
            )
            load_index.cache_clear()

    print("=== self-test PASSED ===")


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
