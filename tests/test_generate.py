"""Offline tests for the generate node. chat is monkeypatched; no Ollama needed.

Evidence numbering and citation-marker parsing moved to test_evidence.py when
those became `meridian.evidence` -- this module now covers only what the node
itself decides: the no-evidence exit, the feedback block, and the iteration
counter it owns.
"""
from meridian import config
from meridian.nodes.generate import generate


def _chunk(marker_text, doc="bert.pdf", page=1, heading="Introduction"):
    return {
        "chunk_id": f"{doc}::p{page}::c0",
        "document_name": doc,
        "page_number": page,
        "section_heading": heading,
        "text": marker_text,
        "score": 0.5,
    }


def test_generate_no_evidence_returns_fallback(monkeypatch):
    # No LLM call should happen when there's no evidence.
    import meridian.nodes.generate as gen_module

    def _boom(*a, **k):
        raise AssertionError("chat() must not be called with no evidence")

    monkeypatch.setattr(gen_module, "chat", _boom)

    result = generate({"question": "q", "retrieved": []})
    assert result["citations"] == []
    assert "couldn't find relevant evidence in the indexed documents" in result["answer"]
    assert result["trace"] == ["generate → no evidence"]


def test_generate_resolves_citations_from_answer_markers(monkeypatch):
    import meridian.nodes.generate as gen_module

    retrieved = [
        _chunk("evidence one", doc="bert.pdf", page=3),
        _chunk("evidence two", doc="t5.pdf", page=7),
    ]
    # Model cites only [2].
    monkeypatch.setattr(gen_module, "chat", lambda *a, **k: "The answer is X [2].")

    result = generate({"question": "q", "retrieved": retrieved})
    assert len(result["citations"]) == 1
    cite = result["citations"][0]
    assert cite["marker"] == 2
    assert cite["document_name"] == "t5.pdf"
    assert cite["page_number"] == 7


def test_generate_recovers_citations_from_a_range(monkeypatch):
    # REGRESSION: the model emits "[1-2]" and both citations must resolve.
    import meridian.nodes.generate as gen_module

    retrieved = [_chunk("one", doc="a.pdf"), _chunk("two", doc="b.pdf", page=2)]
    monkeypatch.setattr(gen_module, "chat", lambda *a, **k: "Both agree [1-2].")

    result = generate({"question": "q", "retrieved": retrieved})
    assert [c["marker"] for c in result["citations"]] == [1, 2]


def test_generate_ignores_out_of_range_markers(monkeypatch):
    import meridian.nodes.generate as gen_module

    retrieved = [_chunk("only one piece of evidence")]
    # Model hallucinates a [5] citation that doesn't exist.
    monkeypatch.setattr(gen_module, "chat", lambda *a, **k: "Answer [5].")

    result = generate({"question": "q", "retrieved": retrieved})
    assert result["citations"] == []


# ---------- the iteration counter, which generate exclusively owns ----------

def test_generate_increments_iterations_from_absent(monkeypatch):
    import meridian.nodes.generate as gen_module

    monkeypatch.setattr(gen_module, "chat", lambda *a, **k: "Answer [1].")
    result = generate({"question": "q", "retrieved": [_chunk("evidence")]})
    assert result["iterations"] == 1


def test_generate_increments_iterations_on_each_attempt(monkeypatch):
    import meridian.nodes.generate as gen_module

    monkeypatch.setattr(gen_module, "chat", lambda *a, **k: "Answer [1].")
    result = generate({"question": "q", "retrieved": [_chunk("evidence")], "iterations": 2})
    assert result["iterations"] == 3


def test_generate_counts_the_no_evidence_attempt_too(monkeypatch):
    # An attempt was still made; leaving the count untouched would let the heal
    # loop believe it has budget it has already spent.
    import meridian.nodes.generate as gen_module

    monkeypatch.setattr(gen_module, "chat", lambda *a, **k: "unused")
    result = generate({"question": "q", "retrieved": [], "iterations": 1})
    assert result["iterations"] == 2


def test_generate_trace_reports_only_its_own_step(monkeypatch):
    # The trace channel is a reducer: a node returns its step, never the history.
    import meridian.nodes.generate as gen_module

    monkeypatch.setattr(gen_module, "chat", lambda *a, **k: "Answer [1].")
    result = generate({"question": "q", "retrieved": [_chunk("evidence")], "trace": ["earlier"]})
    assert len(result["trace"]) == 1
    assert "generate → answer with 1 citation(s)" in result["trace"][0]


# ---------- prompt construction ----------

def test_generate_uses_capped_pool(monkeypatch):
    from meridian.nodes import generate as gen_mod
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
    gen_mod.generate({"question": "q?", "retrieved": retrieved})
    # Only POOL_CAP chunks appear in the prompt
    assert f"[{config.POOL_CAP}]" in seen["user"]
    assert f"[{config.POOL_CAP + 1}]" not in seen["user"]


def test_generate_appends_verify_feedback(monkeypatch):
    from meridian.nodes import generate as gen_mod
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
    }
    gen_mod.generate(state)
    assert "PREVIOUS ATTEMPT FAILED VERIFICATION" in seen["user"]
    assert "1/135" in seen["user"]


def test_generate_no_feedback_block_when_clean(monkeypatch):
    from meridian.nodes import generate as gen_mod
    seen = {}

    def fake_chat(system, user, **kw):
        seen["user"] = user
        return "Answer [1]."

    monkeypatch.setattr(gen_mod, "chat", fake_chat)
    state = {
        "question": "q?",
        "retrieved": [{"chunk_id": "c1", "document_name": "d.pdf", "page_number": 1,
                       "section_heading": "", "text": "evidence", "score": 1.0}],
    }
    gen_mod.generate(state)
    assert "PREVIOUS ATTEMPT" not in seen["user"]
