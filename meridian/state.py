"""The graph's state contract.

Two channels use LangGraph reducers rather than last-write-wins:

- `trace` accumulates. Every node returns only the step(s) it performed and the
  reducer appends them, which is why no node reads the existing trace. Before
  this, all seven nodes open-coded `trace = list(state.get("trace", []))` /
  append / return -- seven chances to drop the history by returning a bare list.
- `retrieved` is pooled by `merge_retrieved`, which keeps the better-scoring
  copy of any chunk seen twice. The heal loop re-enters `search` with new
  queries and must *grow* the evidence pool, never replace it.

`iterations` is deliberately NOT a reducer: it needs to reset per question, and
an accumulating channel cannot be reset. It is instead owned by exactly one
node -- `generate` increments it, nothing else writes it -- which is what makes
the count trustworthy. (It used to be incremented in `search` *and*
pre-incremented in `verify`, with a comment explaining that regeneration
bypassed search's counter.) Every heal path terminates at `generate`, so
counting generations counts attempts.
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class RetrievedChunk(TypedDict, total=False):
    """One pooled evidence chunk. `score` is the FAISS inner product, or 0.0 for
    page-1 chunks boosted in by document scoping."""

    chunk_id: str
    text: str
    page_number: int
    section_heading: str
    document_name: str
    document_title: str
    score: float


class Citation(TypedDict):
    """One resolved `[n]` marker, pointing at the chunk it cites."""

    marker: int
    chunk_id: str
    document_name: str
    page_number: int
    section_heading: str
    score: float | None


def merge_retrieved(
    existing: list[RetrievedChunk], new: list[RetrievedChunk]
) -> list[RetrievedChunk]:
    """Pools evidence across search iterations, de-duplicating on chunk_id and
    keeping the highest score seen for each. Returns score-sorted order, which
    is what `cap_pool` assumes on the way in."""
    by_id: dict[str, RetrievedChunk] = {c["chunk_id"]: c for c in existing}
    for c in new:
        prev = by_id.get(c["chunk_id"])
        if prev is None or c.get("score", 0.0) > prev.get("score", 0.0):
            by_id[c["chunk_id"]] = c
    return sorted(by_id.values(), key=lambda c: c.get("score", 0.0), reverse=True)


class RAGState(TypedDict, total=False):
    question: str

    route: str
    mode_label: str

    sub_questions: list[str]

    retrieved: Annotated[list[RetrievedChunk], merge_retrieved]

    answer: str
    citations: list[Citation]

    critique_clean: bool
    unsupported_claims: list[str]
    iterations: int
    failure_type: str
    verify_feedback: str
    heal_action: str
    retry_queries: list[str]
    verification_warnings: list[str]

    trace: Annotated[list[str], operator.add]
