"""Central configuration for the Adaptive RAG system.

Every tunable knob (model names, iteration cap, chunk sizes, paths) lives here so
there is exactly one place to change behavior. Nothing here reaches for AWS or any
other cloud service — this build is fully local except for Groq LLM calls.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env once at import time so every module sees GROQ_API_KEY.
load_dotenv()

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PAPERS_DIR = DATA_DIR / "papers"          # user drops PDFs here
INDEX_DIR = PROJECT_ROOT / "index"        # persisted FAISS index + metadata
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.json"

# --------------------------------------------------------------------------- #
# LLM (Groq) — used for routing, decompose, generate, critique
# --------------------------------------------------------------------------- #
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# Fast, capable, and cheap on Groq. One model for every LLM step keeps the demo simple.
GROQ_MODEL = "llama-3.3-70b-versatile"
# Deterministic-ish routing/critique matters more than creativity here.
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 1024

# --------------------------------------------------------------------------- #
# Embeddings — fully local sentence-transformers
# --------------------------------------------------------------------------- #
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384                        # all-MiniLM-L6-v2 output dimension

# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
# Spec asked for 512 tokens / 50 overlap, but all-MiniLM-L6-v2 truncates input at
# 256 tokens — a 512-token chunk would be half-embedded (silent truncation). We chose
# 256/50 instead so the embedding vector represents the ENTIRE chunk, giving honest
# retrieval relevance. Cost is ~2x more chunks (trivial at this corpus size). See the
# BUILD_LOG entry for this decision and the deviation from the spec's 512.
CHUNK_SIZE_TOKENS = 256
CHUNK_OVERLAP_TOKENS = 50

# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
# Bumped from 4 to 6 to offset the smaller chunks: each chunk now carries less context,
# so we retrieve a couple more to keep the generate/multi-hop steps well-grounded.
TOP_K = 6                                  # chunks returned per FAISS search

# --------------------------------------------------------------------------- #
# Adaptive routing / self-critique loop
# --------------------------------------------------------------------------- #
# Mode 3 hard cap: total SEARCH passes before we force an exit. This is the
# guardrail that makes the conditional loop terminate.
MAX_ITERATIONS = 3

# Routing labels emitted by the router node.
ROUTE_EASY = "easy"       # Mode 1: no retrieval, answer directly
ROUTE_MEDIUM = "medium"   # Mode 2: single-hop retrieval
ROUTE_HARD = "hard"       # Mode 3: multi-hop + self-critique loop

# Human-readable mode labels for the UI (the demo's headline element).
MODE_LABELS = {
    ROUTE_EASY: "Mode 1 · No Retrieval",
    ROUTE_MEDIUM: "Mode 2 · Single-Hop Retrieval",
    ROUTE_HARD: "Mode 3 · Multi-Hop + Self-Critique",
}
