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


def test_search_node_includes_page_one_for_scoped_query(monkeypatch):
    import src.nodes.search as search_module

    # match_document says the query is about xlnet.pdf.
    monkeypatch.setattr(search_module, "match_document", lambda q: "xlnet.pdf")
    # faiss_search returns only content chunks (no page-1 author block).
    monkeypatch.setattr(
        search_module,
        "faiss_search",
        lambda q, k, document_hint=None: [
            {"chunk_id": "xlnet.pdf::p6::c0", "score": 0.47, "document_name": "xlnet.pdf"}
        ],
    )
    # page_one_chunks returns the author block.
    monkeypatch.setattr(
        search_module,
        "page_one_chunks",
        lambda name: [
            {"chunk_id": "xlnet.pdf::p1::c0", "score": 0.0,
             "document_name": "xlnet.pdf", "page_number": 1, "text": "Zhilin Yang, Zihang Dai"}
        ],
    )

    result = search_module.search_node({"question": "in xlnet who are the authors", "trace": []})
    ids = {c["chunk_id"] for c in result["retrieved"]}
    assert "xlnet.pdf::p1::c0" in ids   # page-1 author block was boosted in
    assert "xlnet.pdf::p6::c0" in ids   # normal semantic hit still present


def test_search_node_no_page_one_for_unscoped_query(monkeypatch):
    import src.nodes.search as search_module

    monkeypatch.setattr(search_module, "match_document", lambda q: None)
    monkeypatch.setattr(
        search_module, "faiss_search",
        lambda q, k, document_hint=None: [{"chunk_id": "a::p2::c0", "score": 0.5}],
    )
    called = {"page_one": False}

    def _page_one(name):
        called["page_one"] = True
        return []

    monkeypatch.setattr(search_module, "page_one_chunks", _page_one)

    search_module.search_node({"question": "some general question", "trace": []})
    assert called["page_one"] is False  # never fetched when no document matched


if __name__ == "__main__":
    test_merge_keeps_higher_scoring_duplicate()
    test_merge_sorts_by_score_descending()
    print("=== non-monkeypatch search tests PASSED ===")
