import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nodes.generate import _cited_indices, _format_evidence, generate


def _chunk(marker_text, doc="bert.pdf", page=1, heading="Introduction"):
    return {
        "chunk_id": f"{doc}::p{page}::c0",
        "document_name": doc,
        "page_number": page,
        "section_heading": heading,
        "text": marker_text,
        "score": 0.5,
    }


def test_format_evidence_numbers_and_labels_each_chunk():
    evidence = _format_evidence([_chunk("first"), _chunk("second", page=2)])
    assert "[1] (bert.pdf p1 · Introduction) first" in evidence
    assert "[2] (bert.pdf p2 · Introduction) second" in evidence


def test_format_evidence_omits_missing_heading():
    chunk = _chunk("body")
    chunk["section_heading"] = ""
    evidence = _format_evidence([chunk])
    assert "·" not in evidence
    assert "[1] (bert.pdf p1) body" in evidence


def test_cited_indices_extracts_valid_markers_sorted_deduped():
    # Out of order, duplicated, and one out-of-range marker.
    assert _cited_indices("Claim [2] and [1] and again [2] and [9]", n=3) == [1, 2]


def test_cited_indices_empty_when_no_markers():
    assert _cited_indices("No citations here at all.", n=3) == []


def test_generate_no_evidence_returns_fallback(monkeypatch):
    # No LLM call should happen when there's no evidence.
    import src.nodes.generate as gen_module

    def _boom(*a, **k):
        raise AssertionError("chat() must not be called with no evidence")

    monkeypatch.setattr(gen_module, "chat", _boom)

    result = generate({"question": "q", "retrieved": [], "trace": []})
    assert result["citations"] == []
    assert "couldn't find relevant evidence in the indexed documents" in result["answer"]
    assert result["trace"] == ["generate → no evidence"]


def test_generate_resolves_citations_from_answer_markers(monkeypatch):
    import src.nodes.generate as gen_module

    retrieved = [
        _chunk("evidence one", doc="bert.pdf", page=3),
        _chunk("evidence two", doc="t5.pdf", page=7),
    ]
    # Model cites only [2].
    monkeypatch.setattr(gen_module, "chat", lambda *a, **k: "The answer is X [2].")

    result = generate({"question": "q", "retrieved": retrieved, "trace": []})
    assert len(result["citations"]) == 1
    cite = result["citations"][0]
    assert cite["marker"] == 2
    assert cite["document_name"] == "t5.pdf"
    assert cite["page_number"] == 7
    assert result["trace"][-1] == "generate → answer with 1 citation(s)"


def test_generate_ignores_out_of_range_markers(monkeypatch):
    import src.nodes.generate as gen_module

    retrieved = [_chunk("only one piece of evidence")]
    # Model hallucinates a [5] citation that doesn't exist.
    monkeypatch.setattr(gen_module, "chat", lambda *a, **k: "Answer [5].")

    result = generate({"question": "q", "retrieved": retrieved, "trace": []})
    assert result["citations"] == []


def test_generate_uses_capped_pool(monkeypatch):
    from src.nodes import generate as gen_mod
    from src import config
    seen = {}
    def fake_chat(system, user, **kw):
        seen["user"] = user
        return "Answer [1]."
    monkeypatch.setattr(gen_mod, "chat", fake_chat)
    retrieved = [
        {"chunk_id": f"c{i}", "document_name": f"d{i % 6}.pdf", "page_number": 1,
         "section_heading": "", "text": f"chunk text {i}", "score": float(100 - i)}
        for i in range(30)
    ]
    result = gen_mod.generate({"question": "q?", "retrieved": retrieved, "trace": []})
    # Only POOL_CAP chunks appear in the prompt
    assert f"[{config.POOL_CAP}]" in seen["user"]
    assert f"[{config.POOL_CAP + 1}]" not in seen["user"]


def test_generate_appends_verify_feedback(monkeypatch):
    from src.nodes import generate as gen_mod
    seen = {}
    def fake_chat(system, user, **kw):
        seen["user"] = user
        return "Corrected answer [1]."
    monkeypatch.setattr(gen_mod, "chat", fake_chat)
    state = {
        "question": "q?",
        "retrieved": [{"chunk_id": "c1", "document_name": "d.pdf", "page_number": 1,
                       "section_heading": "", "text": "evidence", "score": 1.0}],
        "failure_type": "fabrication",
        "verify_feedback": "the answer contains values not present in the evidence: 1/135",
        "trace": [],
    }
    gen_mod.generate(state)
    assert "PREVIOUS ATTEMPT FAILED VERIFICATION" in seen["user"]
    assert "1/135" in seen["user"]


def test_generate_no_feedback_block_when_clean(monkeypatch):
    from src.nodes import generate as gen_mod
    seen = {}
    def fake_chat(system, user, **kw):
        seen["user"] = user
        return "Answer [1]."
    monkeypatch.setattr(gen_mod, "chat", fake_chat)
    state = {
        "question": "q?",
        "retrieved": [{"chunk_id": "c1", "document_name": "d.pdf", "page_number": 1,
                       "section_heading": "", "text": "evidence", "score": 1.0}],
        "trace": [],
    }
    gen_mod.generate(state)
    assert "PREVIOUS ATTEMPT" not in seen["user"]


if __name__ == "__main__":
    test_format_evidence_numbers_and_labels_each_chunk()
    test_format_evidence_omits_missing_heading()
    test_cited_indices_extracts_valid_markers_sorted_deduped()
    test_cited_indices_empty_when_no_markers()
    print("=== non-monkeypatch generate tests PASSED ===")
