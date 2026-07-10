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
    document_title: str


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _tokenizer():
    return _model().tokenizer


def _count_tokens(text: str) -> int:
    return len(_tokenizer().encode(text, add_special_tokens=False))


def _extract_font_spans(page: "fitz.Page") -> list[tuple[float, str]]:
    """Returns (font_size, text) for every text span on the page, in reading order."""
    raw = page.get_text("dict")
    spans: list[tuple[float, str]] = []
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = (span.get("text") or "").strip()
                if text:
                    spans.append((span.get("size", 0.0), text))
    return spans


def _detect_body_font_size(doc: "fitz.Document", sample_pages: int = 5) -> float:
    """Estimates the document's body-text font size by finding the size that
    covers the most total characters across a sample of pages."""
    size_char_counts: dict[float, int] = {}
    n_pages = min(sample_pages, len(doc))
    for page_index in range(n_pages):
        for size, text in _extract_font_spans(doc[page_index]):
            rounded = round(size, 1)
            size_char_counts[rounded] = size_char_counts.get(rounded, 0) + len(text)
    if not size_char_counts:
        return 10.0
    return max(size_char_counts, key=lambda s: size_char_counts[s])


_TITLE_LINE_SIZE_TOLERANCE = 0.85


def _extract_horizontal_title_lines(page: "fitz.Page") -> list[tuple[float, str]]:
    """Returns (max_font_size, merged_text) for each horizontal text line on
    the page, in reading order.

    Two real-world PDF quirks motivate grouping by line instead of returning
    raw spans (as `_extract_font_spans` does):

    1. Preprint PDFs (e.g. arXiv) commonly stamp a rotated sidebar (the
       "arXiv:..." watermark) along the page edge as a single large-font
       span. That stamp is not part of the document's title/heading layout,
       so non-horizontal lines are excluded entirely.
    2. Some papers typeset their title in a small-caps style where the first
       letter of each word is a larger font than the rest of the word (e.g.
       "ALBERT" rendered as spans of size 17 ("A") + size 13.8 ("LBERT")).
       Picking only the single largest *span* size would grab just the
       leading letters and drop the rest. Grouping spans by line and using
       each line's max size keeps the whole line intact.
    """
    raw = page.get_text("dict")
    lines: list[tuple[float, str]] = []
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            direction = line.get("dir", (1.0, 0.0))
            if abs(direction[0]) < 0.99:
                continue  # not (close to) horizontal; skip rotated/vertical text
            texts: list[str] = []
            sizes: list[float] = []
            for span in line.get("spans", []):
                text = (span.get("text") or "")
                if text.strip():
                    texts.append(text)
                    sizes.append(span.get("size", 0.0))
            if texts:
                # Join without inserting spaces: small-caps titles split one
                # word across spans of different sizes (e.g. "A" + "LBERT"),
                # and spans already carry their own internal spacing.
                lines.append((max(sizes), "".join(texts).strip()))
    return lines


def _looks_like_filename(title: str, stem: str) -> bool:
    """True if an embedded metadata title looks like a junk filename rather
    than a real document title. Real titles are almost always multi-word with
    spaces; junk metadata is often a mangled export filename (no spaces, e.g.
    "constitution_pdf2" or the bare filename stem). Trusting such values over
    font-based extraction produces a garbage title, so they're rejected."""
    t = title.strip().lower()
    if not t:
        return True
    if t == stem.lower():
        return True
    # No spaces and looks filename-shaped (underscores/hyphens joining tokens,
    # or a trailing "pdf"/digit suffix) -> treat as a filename, not a title.
    if " " not in t and (("_" in t) or ("-" in t) or t.endswith("pdf")):
        return True
    return False


def _extract_title(doc: "fitz.Document", pdf_path: Path) -> str:
    metadata_title = (doc.metadata or {}).get("title", "").strip()
    if metadata_title and not _looks_like_filename(metadata_title, pdf_path.stem):
        return metadata_title

    if len(doc) > 0:
        lines = _extract_horizontal_title_lines(doc[0])
        if lines:
            max_size = max(size for size, _ in lines)
            threshold = max_size * _TITLE_LINE_SIZE_TOLERANCE
            title_lines = [text for size, text in lines if size >= threshold]
            candidate = " ".join(title_lines).strip()
            if candidate:
                return candidate

    return pdf_path.stem


_HEADING_SIZE_RATIO = 1.08


def _iter_page_blocks_with_headings(
    page: "fitz.Page", body_size: float
) -> list[tuple[bool, str]]:
    """Returns (is_heading, text) for each block on the page, merging spans
    within a block and classifying the block as a heading if its dominant font
    size is meaningfully larger than the document's body-text size."""
    raw = page.get_text("dict")
    results: list[tuple[bool, str]] = []
    for block in raw.get("blocks", []):
        block_text_parts: list[str] = []
        block_sizes: list[float] = []
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = (span.get("text") or "")
                if text.strip():
                    block_text_parts.append(text)
                    block_sizes.append(span.get("size", 0.0))
        block_text = "".join(block_text_parts).strip()
        if not block_text:
            continue
        max_size = max(block_sizes) if block_sizes else 0.0
        is_heading = (
            max_size >= body_size * _HEADING_SIZE_RATIO
            and len(block_text) < 120
        )
        results.append((is_heading, block_text))
    return results


def parse_pdf(pdf_path: Path) -> list[tuple[int, str, str]]:
    segments: list[tuple[int, str, str]] = []
    current_heading = ""
    doc = fitz.open(pdf_path)
    try:
        body_size = _detect_body_font_size(doc)
        for page_index in range(len(doc)):
            page = doc[page_index]
            page_number = page_index + 1
            buffer: list[str] = []
            for is_heading, block_text in _iter_page_blocks_with_headings(page, body_size):
                if is_heading:
                    if buffer:
                        segments.append((page_number, current_heading, "\n".join(buffer)))
                        buffer = []
                    current_heading = block_text
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
    doc = fitz.open(pdf_path)
    try:
        document_title = _extract_title(doc, pdf_path)
    finally:
        doc.close()

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
                    document_title=document_title,
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
            "FAISS index not found. Run `python -m src.ingest` first to build it."
        )
    index = faiss.read_index(str(config.FAISS_INDEX_PATH))
    with open(config.METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = [Chunk(**d) for d in json.load(f)]
    return index, metadata


def _document_match_corpus() -> dict[str, str]:
    """Maps document_name -> a matchable text blob (title + filename stem) for
    every document currently in the loaded index."""
    _, metadata = load_index()
    corpus: dict[str, str] = {}
    for chunk in metadata:
        if chunk.document_name not in corpus:
            stem = Path(chunk.document_name).stem
            corpus[chunk.document_name] = f"{chunk.document_title} {stem}"
    return corpus


_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Common English words that appear incidentally in paper titles/filenames
# (e.g. "Attention Is All You Need", "gpt2.pdf" vs. a query asking "what
# is..."). Left in, these produce false ties/false positives between
# unrelated documents; filtering them out sharpens matching on the
# distinctive, document-specific vocabulary that actually identifies a paper.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "and", "or", "but", "not",
    "no", "do", "does", "did", "how", "what", "which", "who", "whom",
    "this", "that", "these", "those", "it", "its", "as", "by", "from",
    "about", "into", "use", "uses", "used", "using", "can", "will",
    "would", "should", "could", "i", "you", "he", "she", "we", "they",
    "them", "their", "our", "your", "my",
}


# Matches a run of letters/digits allowing internal hyphens (e.g. "gpt-2",
# "text-to-text"), so hyphenated model names survive as a single unit that can
# be normalized to match an unhyphenated filename stem like "gpt2".
_HYPHEN_RUN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)+")


def _tokenize(text: str) -> set[str]:
    text = text.lower()
    tokens = {t for t in _TOKEN_RE.findall(text) if t not in _STOPWORDS}
    # Add de-hyphenated variants of hyphenated runs so "GPT-2" (which
    # otherwise splits into {"gpt", "2"}) also yields "gpt2", matching a
    # filename stem like "gpt2.pdf". This closes a recall gap where hyphenated
    # model names never matched their own unhyphenated document.
    for run in _HYPHEN_RUN_RE.findall(text):
        joined = run.replace("-", "")
        if joined not in _STOPWORDS:
            tokens.add(joined)
    return tokens


def _score_document_match(query_tokens: set[str], doc_tokens: set[str]) -> int:
    return len(query_tokens & doc_tokens)


# Extra weight given to query tokens that match a document's own filename
# stem (e.g. "bert" in "bert.pdf"). Title-word overlap alone is too weak a
# signal in a corpus of vocabulary-similar sibling papers -- e.g. ALBERT's
# title ("A LITE BERT FOR...") literally contains the word "BERT", so a
# BERT-focused query ties bert.pdf and albert.pdf on title words alone. The
# filename stem is the most direct, unambiguous per-document label, so a
# query naming it should count far more than an incidental shared title word.
_STEM_MATCH_WEIGHT = 3

_MATCH_MARGIN = 2


def match_document(text: str) -> Optional[str]:
    corpus = _document_match_corpus()
    if not corpus:
        return None

    query_tokens = _tokenize(text)
    scores: dict[str, int] = {}
    for name, doc_text in corpus.items():
        base = _score_document_match(query_tokens, _tokenize(doc_text))
        stem_tokens = _tokenize(Path(name).stem.replace("_", " "))
        stem_overlap = _score_document_match(query_tokens, stem_tokens)
        scores[name] = base + _STEM_MATCH_WEIGHT * stem_overlap

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if not ranked or ranked[0][1] == 0:
        return None
    if len(ranked) == 1:
        return ranked[0][0]

    best_name, best_score = ranked[0]
    second_score = ranked[1][1]
    if best_score - second_score >= _MATCH_MARGIN:
        return best_name
    return None


def implicated_documents(text: str) -> list[str]:
    """Every indexed document whose FILENAME-STEM tokens overlap the text's
    tokens. Stem-only (never title words): titles share vocabulary across this
    corpus ("Transformer" is in t5's title), so title matching would implicate
    the wrong document. A stem hit ("bert", "gpt2" via de-hyphenation) is the
    unambiguous signal that the user named that document. Used by the router's
    hard->medium downgrade guard."""
    corpus = _document_match_corpus()
    if not corpus:
        return []
    query_tokens = _tokenize(text)
    implicated = []
    for name in corpus:
        stem_tokens = _tokenize(Path(name).stem.replace("_", " "))
        if query_tokens & stem_tokens:
            implicated.append(name)
    return sorted(implicated)


def search(
    query: str, k: Optional[int] = None, document_hint: Optional[str] = None
) -> list[dict]:
    k = k or config.TOP_K
    index, metadata = load_index()
    q = embed_texts([query])

    if document_hint is None:
        fetch_k = min(k, index.ntotal)
        scores, ids = index.search(q, fetch_k)
    else:
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


def page_one_chunks(document_name: str) -> list[dict]:
    """Returns the given document's page-1 chunks (title/author/abstract block)
    as scored result dicts. Used to boost metadata into the candidate set for
    queries scoped to a specific document, where a raw author/title block ranks
    too low semantically to surface on its own."""
    _, metadata = load_index()
    results: list[dict] = []
    for chunk in metadata:
        if chunk.document_name == document_name and chunk.page_number == 1:
            row = asdict(chunk)
            row["score"] = 0.0  # neutral; _merge keeps it, real hits still rank above
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
            n = build_index(docs)
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
    try:
        from src.logging_config import setup_logging
    except ImportError:
        from logging_config import setup_logging  # type: ignore
    setup_logging()

    parser = argparse.ArgumentParser(description="Build the FAISS index from data/documents/.")
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
