"""Offline tests for the state contract's reducers.

`merge_retrieved` used to be `search._merge`, called by the node by hand. It is
now the `retrieved` channel's reducer, so LangGraph applies it -- which is what
lets the heal loop re-enter search and grow the evidence pool automatically.
"""
import operator
from typing import get_args, get_type_hints

from meridian.state import RAGState, merge_retrieved


def test_merge_keeps_higher_scoring_duplicate():
    existing = [{"chunk_id": "a", "score": 0.5}]
    new = [{"chunk_id": "a", "score": 0.8}, {"chunk_id": "b", "score": 0.3}]
    result = merge_retrieved(existing, new)
    ids_to_scores = {c["chunk_id"]: c["score"] for c in result}
    assert ids_to_scores["a"] == 0.8
    assert ids_to_scores["b"] == 0.3


def test_merge_keeps_existing_when_it_scores_higher():
    existing = [{"chunk_id": "a", "score": 0.9}]
    new = [{"chunk_id": "a", "score": 0.2}]
    assert merge_retrieved(existing, new)[0]["score"] == 0.9


def test_merge_sorts_by_score_descending():
    new = [{"chunk_id": "low", "score": 0.1}, {"chunk_id": "high", "score": 0.9}]
    result = merge_retrieved([], new)
    assert [c["chunk_id"] for c in result] == ["high", "low"]


def test_merge_treats_missing_score_as_zero():
    result = merge_retrieved([], [{"chunk_id": "a"}, {"chunk_id": "b", "score": 0.5}])
    assert [c["chunk_id"] for c in result] == ["b", "a"]


def test_merge_from_empty_existing_is_the_reducer_base_case():
    # LangGraph seeds an annotated list channel with [], so the first search
    # round always calls the reducer with no existing evidence.
    new = [{"chunk_id": "a", "score": 0.5}]
    assert merge_retrieved([], new) == new


def test_retrieved_channel_uses_the_merge_reducer():
    hints = get_type_hints(RAGState, include_extras=True)
    assert merge_retrieved in get_args(hints["retrieved"])


def test_trace_channel_accumulates():
    hints = get_type_hints(RAGState, include_extras=True)
    assert operator.add in get_args(hints["trace"])


def test_iterations_is_not_a_reducer_channel():
    # It must stay resettable per question, which an accumulating channel is not.
    hints = get_type_hints(RAGState, include_extras=True)
    assert get_args(hints["iterations"]) == ()
