from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PAPERS_DIR = DATA_DIR / "papers"
INDEX_DIR = PROJECT_ROOT / "index"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.json"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 1024

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# all-MiniLM-L6-v2 truncates at 256 tokens, so chunks are sized to match
# the embedder's actual window instead of the usual 512.
CHUNK_SIZE_TOKENS = 256
CHUNK_OVERLAP_TOKENS = 50

TOP_K = 6

MAX_ITERATIONS = 3

ROUTE_EASY = "easy"
ROUTE_MEDIUM = "medium"
ROUTE_HARD = "hard"

MODE_LABELS = {
    ROUTE_EASY: "Mode 1 · No Retrieval",
    ROUTE_MEDIUM: "Mode 2 · Single-Hop Retrieval",
    ROUTE_HARD: "Mode 3 · Multi-Hop + Self-Critique",
}
