"""The Chunk record and the token-window chunker that produces it."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from meridian import config
from meridian.ingest.embedding import _tokenizer
from meridian.ingest.pdf import extract_title, parse_pdf


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page_number: int
    section_heading: str
    document_name: str
    document_title: str


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
    document_title = extract_title(pdf_path)

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
