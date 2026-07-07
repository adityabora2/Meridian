"""Search node — FAISS retrieval, shared by Mode 2 (single-hop) and Mode 3 (multi-hop).

Behavior depends on what's in state:
  - Mode 2: no sub_questions → search once on the original question.
  - Mode 3: sub_questions present → search once per sub-question and pool the results.

Results are accumulated across passes and de-duplicated by chunk_id, so when the
self-critique loop sends us back here on a later iteration, newly relevant chunks are
added to the evidence pool rather than replacing it.

The iteration counter is bumped here — it is the value the Phase-5 conditional edge reads
to enforce the 3-iteration cap.

Retrieval helpers come from ingest.py (deviation D2 — no separate retriever.py).
"""

from __future__ import annotations

try:
    from src import config
    from src.ingest import search as faiss_search
    from src.state import RAGState
except ImportError:  # running from inside src/
    import config  # type: ignore
    from ingest import search as faiss_search  # type: ignore
    from state import RAGState  # type: ignore


def _merge(existing: list[dict], new: list[dict]) -> list[dict]:
    """Add new chunks to the pool, de-duplicated by chunk_id, keeping the best score."""
    by_id: dict[str, dict] = {c["chunk_id"]: c for c in existing}
    for c in new:
        prev = by_id.get(c["chunk_id"])
        if prev is None or c.get("score", 0.0) > prev.get("score", 0.0):
            by_id[c["chunk_id"]] = c
    # Return sorted by score descending so downstream nodes see the strongest evidence first.
    return sorted(by_id.values(), key=lambda c: c.get("score", 0.0), reverse=True)


def search_node(state: RAGState) -> RAGState:
    """Retrieve chunks for the question or its sub-questions and pool them into state."""
    sub_questions = state.get("sub_questions") or []
    queries = sub_questions if sub_questions else [state["question"]]

    fresh: list[dict] = []
    for q in queries:
        fresh.extend(faiss_search(q, k=config.TOP_K))

    pooled = _merge(state.get("retrieved", []), fresh)

    trace = list(state.get("trace", []))
    n_q = len(queries)
    trace.append(
        f"search ×{n_q} → {len(fresh)} hits, {len(pooled)} pooled "
        f"(iteration {state.get('iterations', 0) + 1})"
    )
    return {
        "retrieved": pooled,
        "iterations": state.get("iterations", 0) + 1,
        "trace": trace,
    }
