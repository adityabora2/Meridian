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


def test_cap_pool_respects_total_and_per_doc_caps():
    from src.nodes.search import cap_pool
    # 14 chunks from docA (scores 14.0..1.0), 3 from docB (0.9..0.7):
    # the pool EXCEEDS POOL_CAP, so the quota+backfill actually bite. Plain
    # top-12 by score would be 12 docA chunks and zero docB; the quota pass
    # takes 4a+3b, backfill fills the rest with the best skipped a-chunks.
    pool = [
        {"chunk_id": f"a{i}", "document_name": "a.pdf", "score": float(14 - i)}
        for i in range(14)
    ] + [
        {"chunk_id": f"b{i}", "document_name": "b.pdf", "score": 0.9 - i * 0.1}
        for i in range(3)
    ]
    capped = cap_pool(pool)
    a_count = sum(1 for c in capped if c["document_name"] == "a.pdf")
    b_count = sum(1 for c in capped if c["document_name"] == "b.pdf")
    assert b_count == 3   # docB's evidence survives (the F2 protection)
    assert a_count == 9   # 4 quota + 5 backfilled
    assert len(capped) == 12  # POOL_CAP


def test_cap_pool_backfills_when_quota_leaves_room():
    from src.nodes.search import cap_pool
    from src import config
    # One doc with 20 chunks: quota selects 4, backfill tops the pool to POOL_CAP.
    pool = [
        {"chunk_id": f"a{i}", "document_name": "a.pdf", "score": float(20 - i)}
        for i in range(20)
    ]
    capped = cap_pool(pool)
    assert len(capped) == config.POOL_CAP
    # Highest-scored chunks win the backfill
    assert capped[0]["chunk_id"] == "a0"


def test_cap_pool_under_cap_unchanged():
    from src.nodes.search import cap_pool
    pool = [{"chunk_id": "x", "document_name": "a.pdf", "score": 1.0}]
    assert cap_pool(pool) == pool


def test_cap_pool_backfill_is_not_document_filtered():
    from src.nodes.search import cap_pool
    # 6 docA + 2 docB + 2 docC chunks = 10 total, under POOL_CAP=12:
    # every chunk must survive, including docA's chunks 5 and 6.
    pool = (
        [{"chunk_id": f"a{i}", "document_name": "a.pdf", "score": float(10 - i)} for i in range(6)]
        + [{"chunk_id": f"b{i}", "document_name": "b.pdf", "score": 3.0 - i} for i in range(2)]
        + [{"chunk_id": f"c{i}", "document_name": "c.pdf", "score": 1.0 - i * 0.1} for i in range(2)]
    )
    capped = cap_pool(pool)
    assert len(capped) == 10
    assert sum(1 for c in capped if c["document_name"] == "a.pdf") == 6


def test_search_node_consumes_retry_queries(monkeypatch):
    from src.nodes import search as search_mod
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
        "trace": [],
    }
    result = search_mod.search_node(state)
    assert calls == ["retry query"]          # retry queries win over sub_questions
    assert result["retry_queries"] == []      # consumed, cleared
    assert result["iterations"] == 2          # still increments


if __name__ == "__main__":
    test_merge_keeps_higher_scoring_duplicate()
    test_merge_sorts_by_score_descending()
    print("=== non-monkeypatch search tests PASSED ===")
