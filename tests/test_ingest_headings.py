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


def test_parse_pdf_detects_realistic_subtle_font_ratio_headings(tmp_path):
    """Regression test for the calibration bug found against real papers
    (bert.pdf, roberta.pdf, t5.pdf): real academic section headings are often
    only ~10% larger than body text (e.g. 12.0pt heading vs 10.9pt body,
    ratio ~1.10), not the dramatic 1.6-2.0x ratios used in the other synthetic
    fixtures above. A threshold calibrated only against exaggerated ratios
    will silently miss these real, subtle headings while still passing every
    test that only uses large ratios.
    """
    pdf = _make_pdf(
        tmp_path,
        [
            ("2 Related Work", 12.0, "Prior work has explored many approaches. " * 10, 10.9),
            ("3 Experiments", 12.0, "We evaluate on several benchmark datasets. " * 10, 10.9),
        ],
    )
    segments = parse_pdf(pdf)
    headings = {heading for _, heading, _ in segments if heading}
    assert any("Related Work" in h for h in headings)
    assert any("Experiments" in h for h in headings)


def test_parse_pdf_does_not_flag_near_body_size_variation_as_heading(tmp_path):
    """Guards against overcorrecting the threshold: minor font-size variation
    close to body size (e.g. 11.1pt vs 10.9pt body, ratio ~1.018 -- the kind
    of noise seen from ligatures/bold/italic runs in real PDFs) must NOT be
    classified as a heading, or ordinary body text would start polluting
    section labels.
    """
    pdf = _make_pdf(
        tmp_path,
        [
            (
                "4 Conclusion",
                12.0,
                "This sentence has a slightly larger emphasized run of text. " * 10,
                10.9,
            ),
        ],
    )
    segments = parse_pdf(pdf)
    headings = {heading for _, heading, _ in segments if heading}
    # The near-body-size emphasized text must not itself become a heading.
    assert not any("emphasized run" in h for h in headings)


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        test_detect_body_font_size_finds_dominant_size(tmp_path)
        test_extract_font_spans_returns_size_text_pairs(tmp_path)
        test_parse_pdf_detects_large_font_as_heading(tmp_path)
        test_parse_pdf_still_works_on_academic_style_headings(tmp_path)
        test_parse_pdf_detects_realistic_subtle_font_ratio_headings(tmp_path)
        test_parse_pdf_does_not_flag_near_body_size_variation_as_heading(tmp_path)
        print("=== all heading-detection tests PASSED ===")
