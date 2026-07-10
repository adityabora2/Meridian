import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nodes.search import _merge


def test_merge_keeps_higher_scoring_duplicate():
    existing = [{"chunk_id": "a", "score": 0.5}]
    new = [{"chunk_id": "a", "score": 0.8}, {"chunk_id": "b", "score": 0.3}]
    result = _merge(existing, new)
    ids_to_scores = {c["chunk_id"]: c["score"] for c in result}
    assert ids_to_scores["a"] == 0.8
    assert ids_to_scores["b"] == 0.3


def test_merge_sorts_by_score_descending():
    existing = []
    new = [{"chunk_id": "low", "score": 0.1}, {"chunk_id": "high", "score": 0.9}]
    result = _merge(existing, new)
    assert [c["chunk_id"] for c in result] == ["high", "low"]


def test_search_node_calls_faiss_search_with_document_hint(monkeypatch):
    import src.nodes.search as search_module

    calls = []

    def fake_faiss_search(q, k, document_hint=None):
        calls.append((q, document_hint))
        return []

    def fake_match_document(text):
        return "bert.pdf" if "bert" in text.lower() else None

    monkeypatch.setattr(search_module, "faiss_search", fake_faiss_search)
    monkeypatch.setattr(search_module, "match_document", fake_match_document)

    state = {"question": "What does BERT use for pretraining?", "trace": []}
    search_module.search_node(state)

    assert calls == [("What does BERT use for pretraining?", "bert.pdf")]


if __name__ == "__main__":
    test_merge_keeps_higher_scoring_duplicate()
    test_merge_sorts_by_score_descending()
    print("=== non-monkeypatch search tests PASSED ===")
