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
