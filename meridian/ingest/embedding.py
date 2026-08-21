"""The embedding model and its tokenizer — loaded once, shared by ingestion
(chunk sizing, index building) and query time (search, coverage checks)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from meridian import config


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _tokenizer():
    return _model().tokenizer


def count_tokens(text: str) -> int:
    return len(_tokenizer().encode(text, add_special_tokens=False))


def embed_texts(texts: list[str]) -> np.ndarray:
    vecs = _model().encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vecs, dtype="float32")
