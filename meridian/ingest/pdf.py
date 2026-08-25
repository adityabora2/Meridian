"""PDF layout parsing: font-size heuristics for titles and headings, and the
page/heading/body segmentation that chunking consumes.

Everything here is font-size based rather than keyed to academic section names,
so it works on general documents (legal, technical) as well as papers.
"""
from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF


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


def extract_title(pdf_path: Path) -> str:
    """Opens the PDF just long enough to read its title."""
    doc = fitz.open(pdf_path)
    try:
        return _extract_title(doc, pdf_path)
    finally:
        doc.close()


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
    """Segments a PDF into (page_number, section_heading, body_text) tuples."""
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
