from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = DATA_DIR / "documents"
INDEX_DIR = PROJECT_ROOT / "index"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.json"

# Overridable so a container can point at a different local model without a
# rebuild. The embedding model is deliberately NOT overridable: it is coupled to
# EMBEDDING_DIM and to the on-disk index, which would silently mismatch.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 1024

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# all-MiniLM-L6-v2 truncates at 256 tokens, so chunks are sized to match
# the embedder's actual window instead of the usual 512.
CHUNK_SIZE_TOKENS = 256
CHUNK_OVERLAP_TOKENS = 50

TOP_K = 6

# Evidence-pool control: generate/verify consume at most POOL_CAP chunks, and
# no single document may crowd out another's evidence (protects N-way
# comparison questions from having one paper's chunks evicted entirely).
POOL_CAP = 12
PER_DOC_CAP = 4

# An "iteration" is one answer attempt -- generate is the only node that counts
# them, so the budget is literally "how many times may this question be answered".
MAX_ITERATIONS = 3

# Medium answers get one heal: the initial attempt (1) plus one retry (2).
MEDIUM_ITERATION_CAP = 2

# Coverage check: a decompose sub-question counts as addressed when some answer
# sentence reaches this cosine similarity (MiniLM embeddings). Tuned against
# the real corpus; too low misses shallow answers, too high forces needless
# regeneration. Kept at 0.45 (not swept over {0.35..0.50}) after task-8 live
# verification: across the 13-query suite it fired exactly once (Q11, mid-loop)
# and correctly forced a fuller regeneration rather than looping a good answer;
# no query's final failure_type was ever "coverage" (the over-trigger failure
# mode), so no evidence justified moving it.
COVERAGE_SIM_THRESHOLD = 0.45

ROUTE_EASY = "easy"
ROUTE_MEDIUM = "medium"
ROUTE_HARD = "hard"
ROUTE_META = "meta"

MODE_LABELS = {
    ROUTE_EASY: "Mode 1 · No Retrieval",
    ROUTE_MEDIUM: "Mode 2 · Single-Hop Retrieval",
    ROUTE_HARD: "Mode 3 · Multi-Hop + Self-Critique",
    ROUTE_META: "Corpus Info",
}


@contextmanager
def index_at(index_dir: Path):
    """Temporarily point the index paths at another directory, restoring them
    on exit even if the body raises.

    The three index paths are derived from one another, so redirecting them by
    hand means three assignments and a three-way restore in a `finally` -- which
    is exactly what the ingest self-test used to open-code. Callers are
    responsible for clearing the index caches on both sides of the swap
    (`meridian.ingest.clear_caches`), since those memoize the old paths' data.
    """
    global INDEX_DIR, FAISS_INDEX_PATH, METADATA_PATH
    previous = (INDEX_DIR, FAISS_INDEX_PATH, METADATA_PATH)
    index_dir = Path(index_dir)
    INDEX_DIR = index_dir
    FAISS_INDEX_PATH = index_dir / "faiss.index"
    METADATA_PATH = index_dir / "metadata.json"
    try:
        yield index_dir
    finally:
        INDEX_DIR, FAISS_INDEX_PATH, METADATA_PATH = previous
