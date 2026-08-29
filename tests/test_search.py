"""Offline tests for the search node. FAISS and the document matcher are
monkeypatched; no index needed.

Pooling moved to `meridian.state.merge_retrieved` (the `retrieved` channel's
reducer) and pool capping to `meridian.evidence` -- see test_state.py and
test_evidence.py. What is left here is the node's own job: choosing queries,
scoping them, and boosting page-1 metadata.
"""


def test_search_node_calls_faiss_search_with_document_hint(monkeypatch):
    import meridian.nodes.search as search_module

    calls = []

    def fake_faiss_search(q, k, document_hint=None):
        calls.append((q, document_hint))
        return []

    def fake_match_document(text):
        return "bert.pdf" if "bert" in text.lower() else None

    monkeypatch.setattr(search_module, "faiss_search", fake_faiss_search)
    monkeypatch.setattr(search_module, "match_document", fake_match_document)

    state = {"question": "What does BERT use for pretraining?"}
    search_module.search_node(state)

    assert calls == [("What does BERT use for pretraining?", "bert.pdf")]


def test_search_node_includes_page_one_for_scoped_query(monkeypatch):
    import meridian.nodes.search as search_module

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

    result = search_module.search_node({"question": "in xlnet who are the authors"})
    ids = {c["chunk_id"] for c in result["retrieved"]}
    assert "xlnet.pdf::p1::c0" in ids   # page-1 author block was boosted in
    assert "xlnet.pdf::p6::c0" in ids   # normal semantic hit still present


def test_search_node_no_page_one_for_unscoped_query(monkeypatch):
    import meridian.nodes.search as search_module

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

    search_module.search_node({"question": "some general question"})
    assert called["page_one"] is False  # never fetched when no document matched


def test_search_node_boosts_each_document_only_once(monkeypatch):
    import meridian.nodes.search as search_module

    monkeypatch.setattr(search_module, "match_document", lambda q: "bert.pdf")
    monkeypatch.setattr(
        search_module, "faiss_search", lambda q, k, document_hint=None: []
    )
    calls = []

    def _page_one(name):
        calls.append(name)
        return []

    monkeypatch.setattr(search_module, "page_one_chunks", _page_one)

    search_module.search_node(
        {"question": "q", "sub_questions": ["about bert one", "about bert two"]}
    )
    assert calls == ["bert.pdf"]  # two sub-questions, one boost


def test_search_node_consumes_retry_queries(monkeypatch):
    from meridian.nodes import search as search_mod
    calls = []

    def fake_search(q, k=None, document_hint=None):
        calls.append(q)
        return []

    monkeypatch.setattr(search_mod, "faiss_search", fake_search)
    monkeypatch.setattr(search_mod, "match_document", lambda q: None)
    state = {
        "question": "original question",
        "sub_questions": ["sub q one", "sub q two"],
        "retry_queries": ["retry query"],
        "retrieved": [],
        "iterations": 1,
    }
    result = search_mod.search_node(state)
    assert calls == ["retry query"]      # retry queries win over sub_questions
    assert result["retry_queries"] == []  # consumed, cleared


def test_search_node_falls_back_to_sub_questions_then_question(monkeypatch):
    from meridian.nodes import search as search_mod
    calls = []
    monkeypatch.setattr(
        search_mod, "faiss_search",
        lambda q, k=None, document_hint=None: calls.append(q) or [],
    )
    monkeypatch.setattr(search_mod, "match_document", lambda q: None)

    search_mod.search_node({"question": "the question", "sub_questions": ["a", "b"]})
    assert calls == ["a", "b"]

    calls.clear()
    search_mod.search_node({"question": "the question"})
    assert calls == ["the question"]


def test_search_node_does_not_write_iterations(monkeypatch):
    # The counter belongs to generate alone; search writing it too was the
    # double-counting hazard this refactor removed.
    from meridian.nodes import search as search_mod
    monkeypatch.setattr(
        search_mod, "faiss_search", lambda q, k=None, document_hint=None: []
    )
    monkeypatch.setattr(search_mod, "match_document", lambda q: None)
    result = search_mod.search_node({"question": "q", "iterations": 1})
    assert "iterations" not in result


def test_search_node_returns_only_fresh_hits(monkeypatch):
    # Pooling is the reducer's job -- the node must not re-merge prior evidence,
    # or the reducer would see (and re-sort) chunks it already holds.
    from meridian.nodes import search as search_mod
    monkeypatch.setattr(
        search_mod, "faiss_search",
        lambda q, k=None, document_hint=None: [{"chunk_id": "new", "score": 0.5}],
    )
    monkeypatch.setattr(search_mod, "match_document", lambda q: None)
    result = search_mod.search_node(
        {"question": "q", "retrieved": [{"chunk_id": "old", "score": 0.9}]}
    )
    assert [c["chunk_id"] for c in result["retrieved"]] == ["new"]
