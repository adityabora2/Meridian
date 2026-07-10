import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fitz

from src.ingest import _extract_title, _looks_like_filename, build_chunks


def test_looks_like_filename_flags_junk_metadata():
    # Real multi-word titles are not filenames.
    assert not _looks_like_filename("Attention Is All You Need", "attention")
    # Bare stem, or filename-shaped junk, is.
    assert _looks_like_filename("attention", "attention")
    assert _looks_like_filename("constitution_pdf2", "constitution")
    assert _looks_like_filename("my-report-final", "report")
    assert _looks_like_filename("document_pdf", "document")
    assert _looks_like_filename("", "anything")


def test_extract_title_ignores_junk_filename_metadata_prefers_font(tmp_path):
    # A PDF whose embedded metadata title is a mangled filename (not a real
    # title) must fall back to the largest-font text on page 1, not trust the
    # junk metadata. This is the real-world US-Constitution case.
    doc = fitz.open()
    page = doc.new_page()
    # Short title so insert_text doesn't run off the page edge at large font.
    page.insert_text((72, 90), "REAL TITLE", fontsize=36)
    rect = fitz.Rect(72, 140, page.rect.width - 72, 400)
    page.insert_textbox(rect, "Body text follows. " * 20, fontsize=10)
    doc.set_metadata({"title": "somefile_pdf2"})
    path = tmp_path / "somefile.pdf"
    doc.save(str(path))
    doc.close()

    reopened = fitz.open(path)
    try:
        title = _extract_title(reopened, path)
        assert "REAL TITLE" in title
        assert "somefile" not in title.lower()
    finally:
        reopened.close()


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


def test_extract_title_ignores_rotated_sidebar_watermark(tmp_path):
    """Regression test for the arXiv-style rotated sidebar watermark bug:
    a large-font rotated text element (e.g. "arXiv:2101.00000v1 [cs.CL]"
    stamped vertically along the page edge) must not be picked as the
    title, even though its font size exceeds the real horizontal title's.
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "THE REAL PAPER TITLE", fontsize=18)
    # Rotated watermark, larger font than the real title, positioned along
    # the right edge of the page the way arXiv stamps its sidebar.
    page.insert_text(
        (page.rect.width - 20, page.rect.height - 100),
        "arXiv:2101.00000v1 [cs.CL] 1 Jan 2021",
        fontsize=24,
        rotate=90,
    )
    rect = fitz.Rect(72, 120, page.rect.width - 72, 300)
    page.insert_textbox(rect, "Body text follows. " * 20, fontsize=10)
    path = tmp_path / "rotated_watermark.pdf"
    doc.save(str(path))
    doc.close()

    reopened = fitz.open(path)
    try:
        title = _extract_title(reopened, path)
        assert "THE REAL PAPER TITLE" in title
        assert "arXiv" not in title
    finally:
        reopened.close()


def test_extract_title_merges_split_font_size_spans_on_same_line(tmp_path):
    """Regression test for the small-caps split-span bug: a title whose
    first letter is typeset in a larger font than the rest of the word
    (e.g. "ALBERT" as "A" at 17pt + "LBERT TEST TITLE" at 14pt, both on
    the same horizontal line) must be returned in full, not truncated to
    just the single largest span ("A").
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "A", fontsize=17)
    page.insert_text((72 + 13, 72), "LBERT TEST TITLE", fontsize=14)
    rect = fitz.Rect(72, 120, page.rect.width - 72, 300)
    page.insert_textbox(rect, "Small body text follows. " * 20, fontsize=10)
    path = tmp_path / "split_span_title.pdf"
    doc.save(str(path))
    doc.close()

    reopened = fitz.open(path)
    try:
        title = _extract_title(reopened, path)
        assert "ALBERT TEST TITLE" in title
        assert title.strip() != "A"
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
        test_looks_like_filename_flags_junk_metadata()
        test_extract_title_ignores_junk_filename_metadata_prefers_font(tmp_path)
        test_extract_title_uses_metadata_when_present(tmp_path)
        test_extract_title_falls_back_to_largest_font_when_metadata_empty(tmp_path)
        test_extract_title_falls_back_to_filename_stem_when_no_text(tmp_path)
        test_extract_title_ignores_rotated_sidebar_watermark(tmp_path)
        test_extract_title_merges_split_font_size_spans_on_same_line(tmp_path)
        test_build_chunks_sets_same_title_on_every_chunk(tmp_path)
        print("=== all title-extraction tests PASSED ===")
