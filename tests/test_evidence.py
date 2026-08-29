"""Offline tests for the evidence contract shared by generate and verify:
pool capping, evidence numbering, and citation-marker parsing.

These used to live in test_generate.py and test_search.py, split across the two
nodes that duplicated the logic. They are one module now because the code is.
"""
from collections import Counter

from meridian import config
from meridian.evidence import (
    cap_pool,
    format_evidence,
    parse_markers,
    resolve_citations,
    strip_markers,
)


def _chunk(text, doc="bert.pdf", page=1, heading="Introduction", score=0.5, cid=None):
    return {
        "chunk_id": cid or f"{doc}::p{page}::c0",
        "document_name": doc,
        "page_number": page,
        "section_heading": heading,
        "text": text,
        "score": score,
    }


# ---------- evidence numbering ----------

def test_format_evidence_numbers_and_labels_each_chunk():
    evidence = format_evidence([_chunk("first"), _chunk("second", page=2)])
    assert "[1] (bert.pdf p1 · Introduction) first" in evidence
    assert "[2] (bert.pdf p2 · Introduction) second" in evidence


def test_format_evidence_omits_missing_heading():
    chunk = _chunk("body")
    chunk["section_heading"] = ""
    evidence = format_evidence([chunk])
    assert "·" not in evidence
    assert "[1] (bert.pdf p1) body" in evidence


# ---------- marker parsing: the formats the model actually emits ----------

def test_parse_markers_single_markers_sorted_and_deduped():
    m = parse_markers("Claim [2] and [1] and again [2].", n_evidence=3)
    assert m.indices == (1, 2)
    assert m.out_of_range == ()
    assert m.malformed is False


def test_parse_markers_comma_list():
    m = parse_markers("Both sources agree [1,3].", n_evidence=4)
    assert m.indices == (1, 3)


def test_parse_markers_comma_list_with_spaces():
    m = parse_markers("Sources agree [1, 2, 4].", n_evidence=4)
    assert m.indices == (1, 2, 4)


def test_parse_markers_expands_a_range():
    # REGRESSION: qwen2.5:7b emits "[5-12]" on long Mode-3 answers. The old
    # r"\[(\d+)\]" parser matched nothing here, so a correctly grounded answer
    # resolved ZERO citations and verify then failed it for "no citations".
    m = parse_markers("The evidence is consistent [5-12].", n_evidence=12)
    assert m.indices == (5, 6, 7, 8, 9, 10, 11, 12)
    assert m.malformed is False


def test_parse_markers_mixed_range_and_single():
    # The other real observed form: "[1-7,9]".
    m = parse_markers("As shown [1-7,9].", n_evidence=10)
    assert m.indices == (1, 2, 3, 4, 5, 6, 7, 9)


def test_parse_markers_en_dash_range():
    m = parse_markers("See [2–4].", n_evidence=5)
    assert m.indices == (2, 3, 4)


# ---------- marker parsing: failure modes ----------

def test_parse_markers_flags_the_n_placeholder():
    m = parse_markers("The model said [n] instead of an index.", n_evidence=3)
    assert m.malformed is True


def test_parse_markers_flags_numbered_placeholder():
    m = parse_markers("Claim [n2].", n_evidence=3)
    assert m.malformed is True


def test_parse_markers_flags_unterminated_range():
    m = parse_markers("Claim [1-].", n_evidence=3)
    assert m.malformed is True


def test_parse_markers_flags_reversed_range():
    m = parse_markers("Claim [7-1].", n_evidence=10)
    assert m.malformed is True


def test_parse_markers_ignores_prose_brackets():
    # "[sic]" is not a citation and must not be reported as a malformed one.
    m = parse_markers("The source says foo [sic] and cites [1].", n_evidence=2)
    assert m.malformed is False
    assert m.indices == (1,)


def test_parse_markers_separates_out_of_range():
    m = parse_markers("Claim [1] and [9].", n_evidence=3)
    assert m.indices == (1,)
    assert m.out_of_range == (9,)


def test_parse_markers_range_partially_out_of_range():
    m = parse_markers("Claim [2-5].", n_evidence=3)
    assert m.indices == (2, 3)
    assert m.out_of_range == (4, 5)


def test_parse_markers_empty_when_no_markers():
    m = parse_markers("No citations here at all.", n_evidence=3)
    assert m.indices == ()
    assert m.any_found is False
    assert m.malformed is False


def test_parse_markers_absurd_range_is_malformed_not_expanded():
    m = parse_markers("Claim [1-999999].", n_evidence=3)
    assert m.malformed is True
    assert m.indices == ()


# ---------- marker stripping ----------

def test_strip_markers_removes_ranges_so_they_are_not_read_as_numbers():
    # If "[1-7,9]" survived into the fabrication check, 1/7/9 would be treated
    # as numeric claims and flagged as values absent from the evidence.
    assert "1" not in strip_markers("Training used the method [1-7,9].")


def test_strip_markers_does_not_fuse_adjacent_words():
    assert strip_markers("word[1]word") == "word word"


# ---------- citation resolution ----------

def test_resolve_citations_maps_markers_to_chunks():
    retrieved = [_chunk("one", doc="bert.pdf", page=3), _chunk("two", doc="t5.pdf", page=7)]
    citations = resolve_citations("The answer is X [2].", retrieved)
    assert len(citations) == 1
    assert citations[0]["marker"] == 2
    assert citations[0]["document_name"] == "t5.pdf"
    assert citations[0]["page_number"] == 7


def test_resolve_citations_drops_out_of_range_markers():
    citations = resolve_citations("Answer [5].", [_chunk("only one")])
    assert citations == []


def test_resolve_citations_recovers_a_ranged_answer():
    # End-to-end version of the regression: an answer citing a range now
    # resolves real citations instead of silently producing none.
    retrieved = [_chunk(f"chunk {i}", cid=f"c{i}", page=i) for i in range(1, 5)]
    citations = resolve_citations("Everything is supported [1-3].", retrieved)
    assert [c["marker"] for c in citations] == [1, 2, 3]


# ---------- pool capping ----------

def test_cap_pool_limits_total_to_pool_cap():
    pooled = [
        _chunk(f"t{i}", doc=f"d{i % 6}.pdf", cid=f"c{i}", score=float(100 - i))
        for i in range(30)
    ]
    assert len(cap_pool(pooled)) == config.POOL_CAP


def test_cap_pool_quota_pass_caps_each_document():
    # With enough cross-document evidence to fill the pool, the quota pass alone
    # decides the selection and no document may exceed PER_DOC_CAP.
    pooled = [
        _chunk("t", doc=f"d{i % 6}.pdf", cid=f"c{i}", score=float(100 - i))
        for i in range(36)
    ]
    selected = cap_pool(pooled)
    counts = Counter(c["document_name"] for c in selected)
    assert max(counts.values()) <= config.PER_DOC_CAP


def test_cap_pool_never_evicts_a_document_entirely():
    # The protection that matters for N-way comparison questions: one document
    # sweeping the top scores must not push another document's evidence out.
    # (The backfill pass may then take that document past PER_DOC_CAP -- the
    # quota is a fairness pass, not a hard ceiling.)
    pooled = [_chunk("hog", doc="hog.pdf", cid=f"h{i}", score=100.0 - i) for i in range(20)]
    pooled += [_chunk("other", doc="other.pdf", cid=f"o{i}", score=0.1 - i * 0.01) for i in range(2)]
    selected = cap_pool(pooled)
    assert len(selected) == config.POOL_CAP
    assert sum(1 for c in selected if c["document_name"] == "other.pdf") == 2


def test_cap_pool_backfills_when_quota_leaves_room():
    # Only one document exists, so the quota alone would return PER_DOC_CAP;
    # the backfill pass must still fill up to POOL_CAP.
    pooled = [_chunk("t", doc="only.pdf", cid=f"c{i}", score=float(100 - i)) for i in range(30)]
    assert len(cap_pool(pooled)) == config.POOL_CAP


def test_cap_pool_is_stable_across_callers():
    # generate and verify must see byte-identical evidence lists, or citation
    # indices resolve against different numbering.
    pooled = [_chunk(f"t{i}", doc=f"d{i % 4}.pdf", cid=f"c{i}", score=float(50 - i)) for i in range(20)]
    assert [c["chunk_id"] for c in cap_pool(pooled)] == [c["chunk_id"] for c in cap_pool(pooled)]


def test_cap_pool_exact_quota_then_backfill_split():
    # 14 chunks from a.pdf (scores 14..1), 3 from b.pdf (0.9..0.7). Plain top-12
    # by score would be 12 a-chunks and zero b. The quota pass takes 4a + 3b,
    # then the backfill fills the remaining 5 slots from the skipped a-chunks.
    pool = [
        {"chunk_id": f"a{i}", "document_name": "a.pdf", "score": float(14 - i)}
        for i in range(14)
    ] + [
        {"chunk_id": f"b{i}", "document_name": "b.pdf", "score": 0.9 - i * 0.1}
        for i in range(3)
    ]
    capped = cap_pool(pool)
    assert len(capped) == config.POOL_CAP
    assert sum(1 for c in capped if c["document_name"] == "b.pdf") == 3
    assert sum(1 for c in capped if c["document_name"] == "a.pdf") == 9


def test_cap_pool_backfill_takes_the_highest_scored_skipped_chunks():
    pool = [
        {"chunk_id": f"a{i}", "document_name": "a.pdf", "score": float(20 - i)}
        for i in range(20)
    ]
    capped = cap_pool(pool)
    assert len(capped) == config.POOL_CAP
    assert capped[0]["chunk_id"] == "a0"


def test_cap_pool_under_cap_is_unchanged():
    pool = [{"chunk_id": "x", "document_name": "a.pdf", "score": 1.0}]
    assert cap_pool(pool) == pool


def test_cap_pool_backfill_is_not_document_filtered():
    # 6 a + 2 b + 2 c = 10 chunks, under POOL_CAP: every chunk must survive,
    # including a.pdf's 5th and 6th, which the quota pass skipped.
    pool = (
        [{"chunk_id": f"a{i}", "document_name": "a.pdf", "score": float(10 - i)} for i in range(6)]
        + [{"chunk_id": f"b{i}", "document_name": "b.pdf", "score": 3.0 - i} for i in range(2)]
        + [{"chunk_id": f"c{i}", "document_name": "c.pdf", "score": 1.0 - i * 0.1} for i in range(2)]
    )
    capped = cap_pool(pool)
    assert len(capped) == 10
    assert sum(1 for c in capped if c["document_name"] == "a.pdf") == 6
