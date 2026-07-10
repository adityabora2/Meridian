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
