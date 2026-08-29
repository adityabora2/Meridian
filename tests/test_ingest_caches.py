"""Offline tests for the index-derived caches and the path/cache coupling.

`match_document` and `implicated_documents` run several times per question, and
each used to walk the entire metadata list (thousands of chunks) and re-tokenize
every document title. The memos below are what removed that. Their correctness
condition is invalidation: a cached corpus outliving the index it came from
would silently scope retrieval to documents that are no longer there.
"""
from meridian import config
from meridian.ingest import clear_caches, using_index
from meridian.ingest import matching, store


def _cache_sizes():
    return {
        "load_index": store.load_index.cache_info().currsize,
        "page_one": store._page_one_rows.cache_info().currsize,
        "corpus": matching._corpus_from_index.cache_info().currsize,
        "tokenize": matching._tokenize.cache_info().currsize,
    }


def test_tokenize_is_memoized_and_returns_an_immutable_set():
    clear_caches()
    text = "How does GPT-2 handle unsupervised multitask learning?"
    first = matching._tokenize(text)
    second = matching._tokenize(text)
    assert first is second                      # served from the memo
    assert isinstance(first, frozenset)         # a caller cannot mutate it
    assert matching._tokenize.cache_info().hits >= 1


def test_tokenize_still_dehyphenates_when_cached():
    clear_caches()
    for _ in range(2):
        tokens = matching._tokenize("Trace the evolution from GPT-2 to GPT-3")
        assert {"gpt2", "gpt3"} <= tokens


def test_clear_caches_empties_every_index_derived_memo():
    matching._tokenize("prime the tokenizer memo")
    assert _cache_sizes()["tokenize"] > 0
    clear_caches()
    assert all(size == 0 for size in _cache_sizes().values()), _cache_sizes()


def test_using_index_flushes_on_entry_and_exit(tmp_path):
    matching._tokenize("prime the tokenizer memo")
    assert _cache_sizes()["tokenize"] > 0

    with using_index(tmp_path / "swapped"):
        # Entry flush: nothing from the previous index may still be memoized.
        assert _cache_sizes()["tokenize"] == 0
        matching._tokenize("primed inside the block")
        assert _cache_sizes()["tokenize"] > 0

    # Exit flush: nothing from the swapped index may leak back out.
    assert _cache_sizes()["tokenize"] == 0


def test_using_index_restores_the_original_paths(tmp_path):
    original = (config.INDEX_DIR, config.FAISS_INDEX_PATH, config.METADATA_PATH)

    with using_index(tmp_path / "swapped"):
        assert config.INDEX_DIR == tmp_path / "swapped"
        assert config.FAISS_INDEX_PATH == tmp_path / "swapped" / "faiss.index"
        assert config.METADATA_PATH == tmp_path / "swapped" / "metadata.json"

    assert (config.INDEX_DIR, config.FAISS_INDEX_PATH, config.METADATA_PATH) == original


def test_using_index_restores_paths_even_when_the_body_raises(tmp_path):
    original = (config.INDEX_DIR, config.FAISS_INDEX_PATH, config.METADATA_PATH)

    try:
        with using_index(tmp_path / "swapped"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    assert (config.INDEX_DIR, config.FAISS_INDEX_PATH, config.METADATA_PATH) == original


def test_match_document_reads_the_corpus_through_the_patchable_seam(monkeypatch):
    # _document_match_corpus stays a thin indirection over the cached walk so
    # tests can substitute a synthetic corpus without touching the index.
    monkeypatch.setattr(
        matching, "_document_match_corpus",
        lambda: {"alpha.pdf": "alpha distinctive title alpha"},
    )
    clear_caches()
    assert matching.match_document("tell me about alpha") == "alpha.pdf"
