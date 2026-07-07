"""Shared graph state for the Adaptive RAG LangGraph.

Every node reads and writes this typed dict. Defining it once here (rather than inline in
graph.py) means each node file can import and type-check against the same contract, and
the field names never drift between nodes.

LangGraph merges each node's returned dict into this state, so nodes return only the keys
they change.
"""

from __future__ import annotations

from typing import TypedDict


class RAGState(TypedDict, total=False):
    # --- input ---
    question: str

    # --- routing ---
    route: str            # "easy" | "medium" | "hard" (config.ROUTE_*)
    mode_label: str       # human-readable label for the UI (config.MODE_LABELS)

    # --- Mode 3 decomposition ---
    sub_questions: list[str]

    # --- retrieval ---
    # Retrieved chunks accumulated across search passes. Each item is the dict returned
    # by ingest.search(): chunk_id, text, page_number, section_heading, document_name, score.
    retrieved: list[dict]

    # --- generation ---
    answer: str
    citations: list[dict]   # subset of retrieved actually cited, for the UI

    # --- self-critique loop (Mode 3) ---
    critique_clean: bool          # True when every claim is supported
    unsupported_claims: list[str] # claims the critique flagged as ungrounded
    iterations: int               # number of SEARCH passes taken (cap = config.MAX_ITERATIONS)

    # --- trace for the UI / debugging ---
    trace: list[str]              # ordered log of which nodes fired
