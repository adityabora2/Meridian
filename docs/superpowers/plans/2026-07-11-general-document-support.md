# General Document Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove academic-paper-specific assumptions from the ingestion pipeline and
LLM prompts, add per-document title extraction, fold in the cross-paper retrieval
contamination fix, and rename `papers` terminology to `documents` throughout, per
the approved spec at
`docs/superpowers/specs/2026-07-11-general-document-support-design.md`.

**Architecture:** Font-size-based heading detection replaces the academic-keyword
regex in `src/ingest.py`. A new `document_title` field (metadata-first, then
largest-font-on-page-1, then filename-stem fallback) is added to every chunk. A new
`match_document()` + `document_hint` parameter on `search()` lets `search_node`
scope a query to one document when it's unambiguously about that document. Router
and decompose prompts become domain-neutral. `data/papers/` becomes
`data/documents/` throughout.

**Tech Stack:** PyMuPDF (`fitz`) — already a dependency, using its `get_text("dict")`
font-size API (not previously used in this codebase) instead of `get_text("blocks")`.
No new dependencies.

## Global Constraints

- No new dependencies — PyMuPDF's `get_text("dict")` is already available via the
  existing `fitz` import.
- The old keyword-based heading regexes (`_HEADING_KEYWORDS`, `_HEADING_NUMBERED`,
  `_looks_like_heading`) are deleted entirely, not kept as a fallback path.
- `document_title` is computed once per document (one value shared across every
  chunk of that document), never per-chunk.
- `match_document()` must return `None` (safe default = whole-corpus search) on any
  ambiguous or no-match case — never guess wrong and filter out the correct
  document. `search()`'s `document_hint` parameter defaults to `None`, and passing
  `None` must produce byte-for-byte identical behavior to today's `search()` calls
  (no regression to Mode 2/3's existing behavior when hint is absent).
- `PLAN.md` and `BUILD_LOG.md` are append-only historical logs — past entries are
  never rewritten. This work gets new entries, not edits to old ones.
- Every existing test file (`tests/test_decompose.py`, `tests/test_critique.py`,
  `tests/test_graph.py`) must continue passing unchanged after this work, since none
  of them depend on the paper-specific heading/title logic being replaced.
- `data/papers/` → `data/documents/`; `config.PAPERS_DIR` → `config.DOCS_DIR`. Every
  reference to the old name in source code (not historical docs) must be updated
  consistently — a stray reference to the old path/name is a bug, not a style nit.

---

### Task 1: Font-size-based heading detection

**Files:**
- Modify: `src/ingest.py:45-96` (deletes `_HEADING_NUMBERED`, `_HEADING_KEYWORDS`,
  `_looks_like_heading`, `_iter_page_blocks`, `parse_pdf`; adds the replacements below)
- Test: `tests/test_ingest_headings.py`

**Interfaces:**
- Consumes: `fitz.Document`/`fitz.Page` (PyMuPDF, already imported as `fitz` at
  `src/ingest.py:11`).
- Produces: `parse_pdf(pdf_path: Path) -> list[tuple[int, str, str]]` — **same
  signature and return shape as today** (list of `(page_number, heading, text)`
  tuples), so `build_chunks` (unchanged, `src/ingest.py:133-151`) keeps working
  without modification. Only the internal heading-detection logic changes.

This task is a drop-in replacement: `parse_pdf`'s callers never change, only its
internals and the private helper functions around it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ingest_headings.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fitz

from src.ingest import _detect_body_font_size, _extract_font_spans, parse_pdf


def _make_pdf(tmp_path: Path, sections: list[tuple[str, int, str, int]]) -> Path:
    """sections: list of (heading_text, heading_fontsize, body_text, body_fontsize)"""
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for heading, h_size, body, b_size in sections:
        page.insert_text((72, y), heading, fontsize=h_size)
        y += h_size + 4
        rect = fitz.Rect(72, y, page.rect.width - 72, y + 200)
        page.insert_textbox(rect, body, fontsize=b_size)
        y += 100
    path = tmp_path / "synthetic_headings.pdf"
    doc.save(str(path))
    doc.close()
    return path


def test_detect_body_font_size_finds_dominant_size(tmp_path):
    pdf = _make_pdf(
        tmp_path,
        [
            ("BIG HEADING ONE", 18, "Body text one. " * 20, 10),
            ("BIG HEADING TWO", 18, "Body text two. " * 20, 10),
        ],
    )
    doc = fitz.open(pdf)
    try:
        body_size = _detect_body_font_size(doc)
        # Body text (10pt) appears far more often than headings (18pt) by character
        # count, so the detected dominant size should be close to 10, not 18.
        assert 9 <= body_size <= 11
    finally:
        doc.close()


def test_extract_font_spans_returns_size_text_pairs(tmp_path):
    pdf = _make_pdf(
        tmp_path,
        [("A HEADING", 18, "some body text here", 10)],
    )
    doc = fitz.open(pdf)
    try:
        spans = _extract_font_spans(doc[0])
        assert any(size >= 17 and "HEADING" in text for size, text in spans)
        assert any(size <= 11 and "body text" in text for size, text in spans)
    finally:
        doc.close()


def test_parse_pdf_detects_large_font_as_heading(tmp_path):
    pdf = _make_pdf(
        tmp_path,
        [
            ("SECTION 7 - TERMINATION", 20, "This clause governs termination. " * 10, 10),
            ("Chapter 4: Troubleshooting", 20, "Follow these steps. " * 10, 10),
        ],
    )
    segments = parse_pdf(pdf)
    headings = {heading for _, heading, _ in segments if heading}
    assert any("TERMINATION" in h or "SECTION 7" in h for h in headings)
    assert any("Troubleshooting" in h or "Chapter 4" in h for h in headings)


def test_parse_pdf_still_works_on_academic_style_headings(tmp_path):
    pdf = _make_pdf(
        tmp_path,
        [
            ("3.2 Results", 16, "Our experiments show strong performance. " * 10, 10),
        ],
    )
    segments = parse_pdf(pdf)
    headings = {heading for _, heading, _ in segments if heading}
    assert any("Results" in h or "3.2" in h for h in headings)


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        test_detect_body_font_size_finds_dominant_size(tmp_path)
        test_extract_font_spans_returns_size_text_pairs(tmp_path)
        test_parse_pdf_detects_large_font_as_heading(tmp_path)
        test_parse_pdf_still_works_on_academic_style_headings(tmp_path)
        print("=== all heading-detection tests PASSED ===")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/apple/Desktop/Projects/aws-rag && source venv/bin/activate && python tests/test_ingest_headings.py`
Expected: `ImportError: cannot import name '_detect_body_font_size' from 'src.ingest'`

- [ ] **Step 3: Replace the heading-detection logic in `src/ingest.py`**

Delete lines 45-71 (`_HEADING_NUMBERED`, `_HEADING_KEYWORDS`, `_looks_like_heading`,
`_iter_page_blocks`) and replace with:

```python
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


_HEADING_SIZE_RATIO = 1.15


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/apple/Desktop/Projects/aws-rag && source venv/bin/activate && python tests/test_ingest_headings.py`
Expected: `=== all heading-detection tests PASSED ===`

- [ ] **Step 5: Run the existing ingest self-test to confirm no regression**

Run: `cd /Users/apple/Desktop/Projects/aws-rag && source venv/bin/activate && python -m src.ingest --self-test`
Expected: `=== self-test PASSED ===`. This synthetic-PDF test uses
`insert_textbox` at a single font size for its body text with numbered headings
("1  Introduction", etc.) also at plain text size — the self-test's assertions
(`tests` at `src/ingest.py:276-284`) check chunk counts and token windows, not
specific heading text, so they should still pass with the new detection logic even
though the synthetic PDF doesn't vary font sizes dramatically. If this fails,
investigate before proceeding — it means the new heading logic broke something the
old one didn't, which needs to be understood, not worked around.

- [ ] **Step 6: Commit**

```bash
git add src/ingest.py tests/test_ingest_headings.py
git commit -m "Replace keyword-based heading detection with font-size detection"
```

---

### Task 2: Per-document title extraction

**Files:**
- Modify: `src/ingest.py` (adds `document_title` to `Chunk`, adds `_extract_title`,
  wires it into `build_chunks`)
- Test: `tests/test_ingest_title.py`

**Interfaces:**
- Consumes: `_extract_font_spans` (Task 1, same file) for the largest-font fallback;
  `fitz.Document.metadata` (PyMuPDF's built-in dict, already used implicitly by
  `fitz.open()`).
- Produces: `_extract_title(doc: "fitz.Document", pdf_path: Path) -> str`. `Chunk`
  gains `document_title: str` as a new field (after `document_name`, so existing
  positional/keyword usage of `Chunk(...)` in `build_chunks` needs the new kwarg
  added, not a positional slot). Every chunk of the same document must have an
  identical `document_title` value.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ingest_title.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fitz

from src.ingest import _extract_title, build_chunks


def test_extract_title_uses_metadata_when_present(tmp_path):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Some Page Content", fontsize=24)
    doc.set_metadata({"title": "The Real Embedded Title"})
    path = tmp_path / "with_metadata.pdf"
    doc.save(str(path))
    doc.close()

    reopened = fitz.open(path)
    try:
        title = _extract_title(reopened, path)
        assert title == "The Real Embedded Title"
    finally:
        reopened.close()


def test_extract_title_falls_back_to_largest_font_when_metadata_empty(tmp_path):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "THE ACTUAL TITLE TEXT", fontsize=22)
    rect = fitz.Rect(72, 120, page.rect.width - 72, 300)
    page.insert_textbox(rect, "Body text follows. " * 20, fontsize=10)
    path = tmp_path / "no_metadata.pdf"
    doc.save(str(path))
    doc.close()

    reopened = fitz.open(path)
    try:
        title = _extract_title(reopened, path)
        assert "THE ACTUAL TITLE TEXT" in title
    finally:
        reopened.close()


def test_extract_title_falls_back_to_filename_stem_when_no_text(tmp_path):
    doc = fitz.open()
    doc.new_page()  # blank page, no extractable text at all
    path = tmp_path / "blank_document.pdf"
    doc.save(str(path))
    doc.close()

    reopened = fitz.open(path)
    try:
        title = _extract_title(reopened, path)
        assert title == "blank_document"
    finally:
        reopened.close()


def test_build_chunks_sets_same_title_on_every_chunk(tmp_path):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "A CONSISTENT TITLE", fontsize=22)
    rect = fitz.Rect(72, 120, page.rect.width - 72, 700)
    page.insert_textbox(rect, "Body content here. " * 100, fontsize=10)
    path = tmp_path / "consistent.pdf"
    doc.save(str(path))
    doc.close()

    chunks = build_chunks(path)
    assert chunks, "expected at least one chunk"
    titles = {c.document_title for c in chunks}
    assert len(titles) == 1, f"expected one consistent title, got {titles}"


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        test_extract_title_uses_metadata_when_present(tmp_path)
        test_extract_title_falls_back_to_largest_font_when_metadata_empty(tmp_path)
        test_extract_title_falls_back_to_filename_stem_when_no_text(tmp_path)
        test_build_chunks_sets_same_title_on_every_chunk(tmp_path)
        print("=== all title-extraction tests PASSED ===")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/apple/Desktop/Projects/aws-rag && source venv/bin/activate && python tests/test_ingest_title.py`
Expected: `ImportError: cannot import name '_extract_title' from 'src.ingest'`

- [ ] **Step 3: Add `document_title` to `Chunk` and implement `_extract_title`**

In `src/ingest.py`, update the `Chunk` dataclass (currently at lines 20-26):

```python
@dataclass
class Chunk:
    chunk_id: str
    text: str
    page_number: int
    section_heading: str
    document_name: str
    document_title: str
```

Add `_extract_title` (place it near `_extract_font_spans`/`_detect_body_font_size`
from Task 1):

```python
def _extract_title(doc: "fitz.Document", pdf_path: Path) -> str:
    metadata_title = (doc.metadata or {}).get("title", "").strip()
    if metadata_title and metadata_title.lower() != pdf_path.stem.lower():
        return metadata_title

    if len(doc) > 0:
        spans = _extract_font_spans(doc[0])
        if spans:
            max_size = max(size for size, _ in spans)
            largest = [text for size, text in spans if size == max_size]
            candidate = " ".join(largest).strip()
            if candidate:
                return candidate

    return pdf_path.stem
```

Update `build_chunks` (currently lines 133-151) to compute the title once and pass
it to every `Chunk`:

```python
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
```

Note: `parse_pdf` (Task 1) opens and closes its own `fitz.Document` internally, so
`build_chunks` now opens the document twice (once here for the title, once inside
`parse_pdf`). This is intentional — keeps `parse_pdf`'s signature and internals from
Task 1 untouched, and PDF opening is cheap relative to embedding/parsing text.

Also update `self_test`'s assertion at `src/ingest.py:278` (currently
`assert all(c.chunk_id and c.document_name == "synthetic.pdf" for c in chunks)`) to
also check the new field:

```python
        assert all(
            c.chunk_id and c.document_name == "synthetic.pdf" and c.document_title
            for c in chunks
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/apple/Desktop/Projects/aws-rag && source venv/bin/activate && python tests/test_ingest_title.py`
Expected: `=== all title-extraction tests PASSED ===`

- [ ] **Step 5: Run the ingest self-test again to confirm no regression**

Run: `cd /Users/apple/Desktop/Projects/aws-rag && source venv/bin/activate && python -m src.ingest --self-test`
Expected: `=== self-test PASSED ===`

- [ ] **Step 6: Commit**

```bash
git add src/ingest.py tests/test_ingest_title.py
git commit -m "Add per-document title extraction (metadata, then largest-font, then filename)"
```

---

### Task 3: Cross-document retrieval scoping (`match_document` + `document_hint`)

**Files:**
- Modify: `src/ingest.py` (adds `match_document`, adds `document_hint` param to
  `search`)
- Modify: `src/nodes/search.py` (computes and passes `document_hint` per query)
- Test: `tests/test_ingest_match_document.py`, `tests/test_search.py`

**Interfaces:**
- Consumes: `load_index()` (existing, `src/ingest.py:203-214`, returns
  `(faiss_index, list[Chunk])`); `Chunk.document_title`, `Chunk.document_name` (Task
  2).
- Produces: `match_document(text: str) -> str | None` — returns a `document_name`
  string or `None`. `search(query: str, k: Optional[int] = None, document_hint:
  str | None = None) -> list[dict]` — same return shape as today (list of dicts with
  a `score` key added), `document_hint=None` must behave identically to calling
  `search(query, k)` today (no behavior change for existing callers that don't pass
  it). `search_node(state: RAGState) -> RAGState` (existing signature, unchanged) —
  internals now call `match_document` per query.

- [ ] **Step 1: Write the failing test for `match_document`**

Create `tests/test_ingest_match_document.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import _score_document_match, match_document


def test_score_document_match_counts_token_overlap():
    query_tokens = {"how", "does", "bert", "handle", "positional", "encoding"}
    doc_tokens = {"bert", "pretraining", "deep", "bidirectional", "transformers"}
    score = _score_document_match(query_tokens, doc_tokens)
    assert score >= 1  # "bert" overlaps


def test_match_document_returns_clear_winner(monkeypatch):
    fake_docs = {
        "bert.pdf": "bert pretraining deep bidirectional transformers language understanding",
        "t5.pdf": "exploring limits transfer learning unified text to text transformer",
    }

    import src.ingest as ingest_module

    monkeypatch.setattr(ingest_module, "_document_match_corpus", lambda: fake_docs)

    result = match_document("How does BERT handle positional encoding?")
    assert result == "bert.pdf"


def test_match_document_returns_none_when_ambiguous(monkeypatch):
    fake_docs = {
        "doc_a.pdf": "machine learning transformer attention model",
        "doc_b.pdf": "machine learning transformer attention model",
    }

    import src.ingest as ingest_module

    monkeypatch.setattr(ingest_module, "_document_match_corpus", lambda: fake_docs)

    result = match_document("Tell me about the machine learning transformer model")
    assert result is None


def test_match_document_returns_none_for_generic_cross_document_question(monkeypatch):
    fake_docs = {
        "bert.pdf": "bert pretraining deep bidirectional transformers",
        "t5.pdf": "exploring limits transfer learning text to text transformer",
    }

    import src.ingest as ingest_module

    monkeypatch.setattr(ingest_module, "_document_match_corpus", lambda: fake_docs)

    result = match_document("Compare how different papers approach pretraining objectives")
    assert result is None


if __name__ == "__main__":
    test_score_document_match_counts_token_overlap()
    print("=== test_score_document_match_counts_token_overlap PASSED ===")
```

Note: the three `match_document` tests use `monkeypatch`, which requires `pytest`.
This repo has no pytest in `requirements.txt` (confirmed: every other test file uses
plain `assert` + a `__main__` block). Since `match_document` needs to look up
per-document text from the loaded index (not something easily hand-built without
either pytest's `monkeypatch` fixture or a manual monkeypatch), install `pytest` as
a dev-only tool for running this one test file, matching how Playwright was
previously installed as a one-time verification tool without being added to
`requirements.txt`. Run these three via `pytest`, and keep the fourth
(`test_score_document_match_counts_token_overlap`) runnable via plain `python` too,
as shown in the `__main__` block above (it needs no monkeypatching).

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/apple/Desktop/Projects/aws-rag && source venv/bin/activate && pip install pytest && python -m pytest tests/test_ingest_match_document.py -v`
Expected: `ImportError` / collection error — `_score_document_match` and
`match_document` don't exist yet in `src.ingest`.

- [ ] **Step 3: Implement `match_document` in `src/ingest.py`**

Add near `search()`:

```python
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


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _score_document_match(query_tokens: set[str], doc_tokens: set[str]) -> int:
    return len(query_tokens & doc_tokens)


_MATCH_MARGIN = 2


def match_document(text: str) -> Optional[str]:
    corpus = _document_match_corpus()
    if not corpus:
        return None

    query_tokens = _tokenize(text)
    scores = {
        name: _score_document_match(query_tokens, _tokenize(doc_text))
        for name, doc_text in corpus.items()
    }

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
```

This file already imports `re` (line 5) and `Optional` (line 9), so no new imports
are needed.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/apple/Desktop/Projects/aws-rag && source venv/bin/activate && python -m pytest tests/test_ingest_match_document.py -v`
Expected: `4 passed`

Also run the non-pytest fourth test directly to confirm it works standalone too:
Run: `cd /Users/apple/Desktop/Projects/aws-rag && source venv/bin/activate && python tests/test_ingest_match_document.py`
Expected: `=== test_score_document_match_counts_token_overlap PASSED ===`

- [ ] **Step 5: Add `document_hint` to `search()`**

Modify `search()` in `src/ingest.py` (currently lines 217-230):

```python
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
```

- [ ] **Step 6: Write the failing test for `search_node`'s use of `document_hint`**

Create `tests/test_search.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nodes.search import _merge


def test_merge_keeps_higher_scoring_duplicate():
    existing = [{"chunk_id": "a", "score": 0.5}]
    new = [{"chunk_id": "a", "score": 0.8}, {"chunk_id": "b", "score": 0.3}]
    result = _merge(existing, new)
    ids_to_scores = {c["chunk_id"]: c["score"] for c in result}
    assert ids_to_scores["a"] == 0.8
    assert ids_to_scores["b"] == 0.3


def test_merge_sorts_by_score_descending():
    existing = []
    new = [{"chunk_id": "low", "score": 0.1}, {"chunk_id": "high", "score": 0.9}]
    result = _merge(existing, new)
    assert [c["chunk_id"] for c in result] == ["high", "low"]


def test_search_node_calls_faiss_search_with_document_hint(monkeypatch):
    import src.nodes.search as search_module

    calls = []

    def fake_faiss_search(q, k, document_hint=None):
        calls.append((q, document_hint))
        return []

    def fake_match_document(text):
        return "bert.pdf" if "bert" in text.lower() else None

    monkeypatch.setattr(search_module, "faiss_search", fake_faiss_search)
    monkeypatch.setattr(search_module, "match_document", fake_match_document)

    state = {"question": "What does BERT use for pretraining?", "trace": []}
    search_module.search_node(state)

    assert calls == [("What does BERT use for pretraining?", "bert.pdf")]


if __name__ == "__main__":
    test_merge_keeps_higher_scoring_duplicate()
    test_merge_sorts_by_score_descending()
    print("=== non-monkeypatch search tests PASSED ===")
```

- [ ] **Step 7: Run test to verify the new test fails**

Run: `cd /Users/apple/Desktop/Projects/aws-rag && source venv/bin/activate && python -m pytest tests/test_search.py -v`
Expected: `test_search_node_calls_faiss_search_with_document_hint` FAILS —
`search_module` has no attribute `match_document` yet, and `faiss_search` isn't
called with a `document_hint` kwarg yet.

- [ ] **Step 8: Update `search_node` in `src/nodes/search.py`**

Replace the full file:

```python
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
```

- [ ] **Step 9: Run test to verify it passes**

Run: `cd /Users/apple/Desktop/Projects/aws-rag && source venv/bin/activate && python -m pytest tests/test_search.py -v`
Expected: `3 passed`

Also confirm the non-monkeypatch subset runs standalone:
Run: `cd /Users/apple/Desktop/Projects/aws-rag && source venv/bin/activate && python tests/test_search.py`
Expected: `=== non-monkeypatch search tests PASSED ===`

- [ ] **Step 10: Commit**

```bash
git add src/ingest.py src/nodes/search.py tests/test_ingest_match_document.py tests/test_search.py
git commit -m "Add cross-document retrieval scoping via match_document + document_hint"
```

---

### Task 4: Domain-agnostic router and decompose prompts

**Files:**
- Modify: `src/nodes/router.py:15-34`
- Modify: `src/nodes/decompose.py:13-21`
- Modify: `src/nodes/generate.py:42` (the stray "papers" string, per spec Change 6)

**Interfaces:**
- No signature changes anywhere in this task — `route_question`, `decompose`, and
  `generate` all keep their exact existing signatures. Only the string constants
  (`_SYSTEM` prompts, one error-message string) change.

This task has no new automated test — the router/decompose prompts were already
covered by `tests/test_decompose.py` (parsing logic, unaffected by prompt wording)
and Phase 3's live routing spot-checks. Correctness here is judged by re-running the
existing offline tests (must still pass, since they don't depend on prompt wording)
plus a live routing/decompose spot-check in this task.

- [ ] **Step 1: Rewrite `router.py`'s system prompt**

Replace `_SYSTEM` in `src/nodes/router.py` (currently lines 15-34):

```python
_SYSTEM = """You are a query-complexity router for a document Q&A system over an \
indexed collection of documents.

Classify the user's question into exactly one complexity level:

- easy: General knowledge or definitional questions a strong LLM can answer correctly \
WITHOUT looking at any document. No retrieval needed.
  Examples: "What does an acronym like RAG stand for?", "What is a vector database?"

- medium: The answer requires grounding in the documents, but a SINGLE focused search \
will surface it. One fact or concept, findable in one place.
  Examples: "What does document X say about topic Y?", \
"Which method does the source material describe for doing Z?"

- hard: The question is multi-hop or cross-document. Answering it needs several \
searches and evidence chained across sub-questions.
  Examples: "How does the approach in one document differ from another, and what do \
they share?", "Compare how several sources each handle the same underlying problem."

Respond with ONLY one word: easy, medium, or hard. No punctuation, no explanation."""
```

- [ ] **Step 2: Rewrite `decompose.py`'s system prompt**

Replace `_SYSTEM` in `src/nodes/decompose.py` (currently lines 13-21):

```python
_SYSTEM = """You break down a hard, multi-hop question about a collection of indexed \
documents into 2-3 focused sub-questions that can each be answered by a SINGLE, \
independent document search.

Rules:
- Produce 2 or 3 sub-questions, never 1 and never more than 3.
- Each sub-question must be self-contained and independently searchable.
- Output ONLY the sub-questions, one per line.
- No numbering, no bullets, no explanation, no preamble."""
```

- [ ] **Step 3: Fix the stray "papers" string in `generate.py`**

In `src/nodes/generate.py`, line 42:

```python
                "answer": "I couldn't find relevant evidence in the indexed papers to answer this.",
```

becomes:

```python
                "answer": "I couldn't find relevant evidence in the indexed documents to answer this.",
```

- [ ] **Step 4: Re-run existing offline tests to confirm no regression**

Run: `cd /Users/apple/Desktop/Projects/aws-rag && source venv/bin/activate && python tests/test_decompose.py && python tests/test_critique.py && python tests/test_graph.py`
Expected: all three print their `=== ... PASSED ===` lines — none of these test the
prompt wording directly, only the parsing/branching logic, so they must be
unaffected.

- [ ] **Step 5: Live spot-check the router and decompose against the real corpus**

Run:

```bash
cd /Users/apple/Desktop/Projects/aws-rag && source venv/bin/activate && python -c "
from src.nodes.router import route_question
from src.nodes.decompose import decompose

for q in [
    'What is a vector database?',
    'According to the BERT paper, what pretraining tasks does BERT use?',
    'Compare how BERT and the Transformer paper each handle positional information, and explain any tradeoffs.',
]:
    r = route_question({'question': q, 'trace': []})
    print(repr(q[:60]), '->', r.get('route'))

result = decompose({
    'question': 'Compare how BERT and the Transformer paper each handle positional information, and explain any tradeoffs.',
    'trace': [],
})
print('Sub-questions:', result['sub_questions'])
"
```

Expected: the three questions still route to `easy`, `medium`, `hard` respectively
(same behavior as before the prompt rewrite — the rewrite changes wording, not the
underlying easy/medium/hard distinction), and `decompose` still returns 2-3 sensible
sub-questions for the comparison question.

- [ ] **Step 6: Commit**

```bash
git add src/nodes/router.py src/nodes/decompose.py src/nodes/generate.py
git commit -m "Rewrite router/decompose prompts to be domain-agnostic"
```

---

### Task 5: Rename `papers` to `documents` throughout the codebase

**Files:**
- Modify: `src/config.py:12` (`PAPERS_DIR` → `DOCS_DIR`)
- Modify: `src/ingest.py` (every reference to `PAPERS_DIR`, `papers_dir` parameter
  name, "papers"/"PDFs" user-facing strings)
- Modify: any other file referencing `config.PAPERS_DIR` (search required, see Step
  1)
- Rename: `data/papers/` directory → `data/documents/` (including moving the 10
  existing PDF files)

**Interfaces:**
- `config.DOCS_DIR` replaces `config.PAPERS_DIR` — same type (`Path`), same
  meaning, new name. Every internal caller must be updated in the same commit (no
  partial rename, no backwards-compatible alias — this is a rename, not a
  deprecation).

- [ ] **Step 1: Find every reference to the old name**

Run: `cd /Users/apple/Desktop/Projects/aws-rag && grep -rn "PAPERS_DIR\|papers_dir\|data/papers" --include="*.py" .`

Expected output includes at least:
- `src/config.py:12` — the definition
- `src/ingest.py:168` — `papers_dir = papers_dir or config.PAPERS_DIR`
- `src/ingest.py:165` — `def build_index(papers_dir: Optional[Path] = None) -> int:`
- `src/ingest.py:169` — `pdfs = sorted(Path(papers_dir).glob("*.pdf"))`
- `src/ingest.py:171-173` — the `FileNotFoundError` message
- `src/ingest.py:271` — `papers = Path(tmp) / "papers"` (inside `self_test`)
- `src/ingest.py:319` — the argparse description string

Read the full grep output and confirm every hit is accounted for in the steps
below — if the grep finds a reference not listed above, add a step to fix it before
proceeding.

- [ ] **Step 2: Rename `config.PAPERS_DIR` to `config.DOCS_DIR`**

In `src/config.py`, line 12:

```python
DOCS_DIR = DATA_DIR / "documents"
```

(was `PAPERS_DIR = DATA_DIR / "papers"`)

- [ ] **Step 3: Update `src/ingest.py`'s references**

Update `build_index` (currently lines 165-173):

```python
def build_index(docs_dir: Optional[Path] = None) -> int:
    import faiss

    docs_dir = docs_dir or config.DOCS_DIR
    pdfs = sorted(Path(docs_dir).glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"No PDFs found in {docs_dir}. Drop the documents there and re-run ingestion."
        )
```

Update `self_test`'s local variable (currently line 271, inside the `with
tempfile.TemporaryDirectory()` block):

```python
        docs = Path(tmp) / "documents"
        pdf = docs / "synthetic.pdf"
```

(and update the one other use of the old `papers` local variable name later in the
same function, currently `n = build_index(papers)`, to `n = build_index(docs)`)

Update the argparse description (currently line 319):

```python
    parser = argparse.ArgumentParser(description="Build the FAISS index from data/documents/.")
```

- [ ] **Step 4: Move the actual PDF files**

Run:

```bash
cd /Users/apple/Desktop/Projects/aws-rag
mkdir -p data/documents
git mv data/papers/*.pdf data/documents/ 2>/dev/null || mv data/papers/*.pdf data/documents/
rmdir data/papers 2>/dev/null || true
ls data/documents/
```

Expected: all 10 PDFs now listed under `data/documents/`; `data/papers/` no longer
exists (or is empty and removed). Note: `data/papers/*.pdf` is gitignored per
`.gitignore`, so `git mv` may report the files as untracked — a plain `mv` fallback
handles this; either way the files end up in the new location.

- [ ] **Step 5: Re-run ingestion against the new location**

Run: `cd /Users/apple/Desktop/Projects/aws-rag && source venv/bin/activate && python -m src.ingest`
Expected: `Indexed 1490 chunks from 10 PDF(s).` (same chunk count as before, since
it's the same 10 files, just relocated — if the count differs, the heading-detection
or title-extraction changes from Tasks 1-2 changed chunking behavior in an
unexpected way and need investigation before proceeding).

- [ ] **Step 6: Run the ingest self-test to confirm the rename didn't break anything**

Run: `cd /Users/apple/Desktop/Projects/aws-rag && source venv/bin/activate && python -m src.ingest --self-test`
Expected: `=== self-test PASSED ===`

- [ ] **Step 7: Commit**

```bash
git add -A src/config.py src/ingest.py data/documents data/papers
git commit -m "Rename data/papers -> data/documents and PAPERS_DIR -> DOCS_DIR"
```

---

### Task 6: Full live verification across the renamed, generalized pipeline

**Files:**
- None created/modified — this task runs the full system live and records results
  in `BUILD_LOG.md` and `PLAN.md`.

**Interfaces:**
- Consumes: everything from Tasks 1-5 — the rebuilt index (Task 5, Step 5), the
  compiled graph (`src/graph.py`, unmodified by this plan), `app.py` (unmodified).

This is a manual verification task (live LLM + real corpus), matching the pattern
used for every previous phase's live verification in this project.

- [ ] **Step 1: Measure the new heading-detection accuracy**

Run:

```bash
cd /Users/apple/Desktop/Projects/aws-rag && source venv/bin/activate && python -c "
import json
with open('index/metadata.json') as f:
    chunks = json.load(f)

from collections import Counter
headings = Counter(c['section_heading'] for c in chunks)
total = len(chunks)
academic_generic = sum(
    count for heading, count in headings.items()
    if heading.strip().lower() in ('abstract', 'references', 'acknowledgments', 'acknowledgements', '')
)
print(f'Total chunks: {total}')
print(f'Chunks with generic/empty heading: {academic_generic} ({100*academic_generic/total:.1f}%)')
print()
print('Top 15 headings by frequency:')
for heading, count in headings.most_common(15):
    print(f'  {count:4d}  {heading!r}')
"
```

Record the output. Compare the generic-heading percentage against the
previously-measured 86% (from the production audit, entry logged in BUILD_LOG
entry 12) — this is the direct, quantified evidence of whether font-size detection
actually improved heading quality. Whatever the number is, record it honestly in
BUILD_LOG — do not round favorably or omit an unfavorable result.

- [ ] **Step 2: Spot-check `match_document` and document-scoped search on 3 real documents**

Run:

```bash
cd /Users/apple/Desktop/Projects/aws-rag && source venv/bin/activate && python -c "
from src.ingest import match_document, search

queries = [
    'How does BERT use masked language modeling?',
    'What is the T5 text-to-text framework?',
    'How does ELECTRA use replaced token detection?',
]
for q in queries:
    hint = match_document(q)
    print(f'{q!r} -> matched document: {hint}')
    results = search(q, k=3, document_hint=hint)
    for r in results:
        print(f'    {r[\"document_name\"]} p{r[\"page_number\"]} (score={r[\"score\"]:.3f})')
"
```

Expected: each query's `match_document` result names the correct paper
(`bert.pdf`, `t5.pdf`, `electra.pdf` respectively), and every returned result's
`document_name` matches that hint (proving the filter is actually active, not just
computed and ignored).

- [ ] **Step 3: Run all three modes end-to-end through the full graph**

Run:

```bash
cd /Users/apple/Desktop/Projects/aws-rag && source venv/bin/activate && python -c "
from src.graph import build_graph

g = build_graph()

for q in [
    'What is a vector database?',
    'According to the BERT paper, what pretraining tasks does BERT use?',
    'Compare how BERT and the Transformer paper each handle positional information, and explain any tradeoffs.',
]:
    result = g.invoke({'question': q, 'trace': []})
    print('Question:', q)
    print('Route:', result.get('route'))
    print('Citations:', [(c['document_name'], c['page_number']) for c in result.get('citations', [])])
    print('Iterations:', result.get('iterations'))
    print()
"
```

Expected: same routing (`easy`/`medium`/`hard`) as every prior live verification in
this project; Mode 2's citation is into `bert.pdf`; Mode 3's citations span both
`bert.pdf` and `attention_is_all_you_need.pdf` (confirming the document-scoping fix
didn't break legitimate cross-document comparison); `Iterations` for Mode 3 is
between 1 and 3.

- [ ] **Step 4: Update `PLAN.md`**

Add a note under the existing Phase 4/5 sections or as a new dated addendum
(whichever reads more naturally given the current PLAN.md structure) documenting:
heading-detection accuracy before/after (86% generic → the Step 1 measurement),
confirmation that document-scoping is active and correct (Step 2), and the
end-to-end mode verification (Step 3). Reference
`docs/superpowers/specs/2026-07-11-general-document-support-design.md` as the spec
this addendum implements.

- [ ] **Step 5: Add a `BUILD_LOG.md` entry**

Append a new entry (next sequential number) documenting: what was built across all 6
tasks (font-size heading detection, title extraction, cross-document scoping,
domain-agnostic prompts, the papers→documents rename), why (the user's explicit
push-back on the project being scoped to research papers only, combined with the
previously-paused cross-paper-contamination audit finding), the exact before/after
heading-detection numbers from Step 1, and the live verification results from Steps
2-3.

- [ ] **Step 6: Commit**

```bash
git add PLAN.md BUILD_LOG.md
git commit -m "Complete general-document-support: verified live, heading accuracy measured"
```

---

## Self-Review Notes

- **Spec coverage:** Change 1 (font-size heading detection) → Task 1. Change 2
  (title extraction) → Task 2. Change 3 (cross-document contamination fix) → Task
  3. Change 4 (domain-agnostic prompts) → Task 4. Change 5 (papers→documents
  rename) → Task 5. Change 6 (generate.py stray string) → folded into Task 4, Step
  3. Testing approach's "generalization proof" (ingesting a genuinely non-paper
  PDF) is **not** a separate task — noted as a gap below.
- **Placeholder scan:** no TBD/TODO; every step has complete runnable code; the one
  tunable constant (`_HEADING_SIZE_RATIO = 1.15`, `_MATCH_MARGIN = 2`) are given
  concrete starting values with tests that validate the behavior they produce,
  rather than left as "tune this later" — if Task 6's live verification shows these
  values need adjustment, that's a normal implementation-time tuning step, not a
  missing requirement.
- **Type consistency:** `parse_pdf(pdf_path: Path) -> list[tuple[int, str, str]]`
  signature preserved exactly from Task 1 through every later task. `Chunk`'s new
  `document_title: str` field is threaded consistently: added in Task 2, read by
  `match_document`/`_document_match_corpus` in Task 3. `search(query, k,
  document_hint)` signature introduced in Task 3 is used identically by
  `search_node` in the same task and by Task 6's live verification script.
  `config.PAPERS_DIR` → `config.DOCS_DIR` rename in Task 5 is applied consistently
  to every call site found by Task 5 Step 1's grep.
- **Gap identified during self-review:** the spec's testing section calls for
  "ingest at least one genuinely non-research-paper PDF... to confirm heading
  detection, title extraction, and Q&A all work correctly on it" as concrete proof
  of generalization. This plan's Task 6 verifies against the existing 10 research
  papers only (they're still valid PDFs, but don't prove the paper-specific
  assumption is actually gone). This is intentionally left as a follow-up rather
  than a blocking task here, since the plan has no access to a specific non-paper
  PDF to test against — **the user should supply one** (a manual, report, or any
  other non-academic PDF) after this plan's tasks land, and a quick manual
  `python -m src.ingest` + a live query against it would be the concrete
  confirmation. Flagging this explicitly rather than silently dropping it.
