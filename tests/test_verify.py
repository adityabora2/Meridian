"""Offline tests for the verification gate. chat and embed_texts are
monkeypatched; no Ollama or network needed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.nodes import verify as verify_mod
from src.nodes.verify import (
    _canon_number,
    _check_citations,
    _check_numbers,
    _parse_support,
    verify,
)


def _evidence(texts):
    return [
        {
            "chunk_id": f"c{i}",
            "document_name": "doc.pdf",
            "page_number": 1,
            "section_heading": "",
            "text": t,
            "score": 1.0,
        }
        for i, t in enumerate(texts, start=1)
    ]


# ---------- number canonicalization ----------

def test_canon_number_folds_commas_and_magnitudes():
    assert _canon_number("4,000") == "4000"
    assert _canon_number("175B") == "175billion"
    assert _canon_number("175 billion") == "175billion"
    assert _canon_number("93.3%") == "93.3%"
    assert _canon_number("1/4") == "1/4"


# ---------- citation check ----------

def test_check_citations_ok():
    ok, fb = _check_citations("Adam is used [1] with warmup [2].", 3)
    assert ok

def test_check_citations_out_of_range():
    ok, fb = _check_citations("Adam is used [7].", 3)
    assert not ok

def test_check_citations_malformed_marker():
    ok, fb = _check_citations("PaLM outperforms [n14] and [n].", 3)
    assert not ok

def test_check_citations_none_present():
    ok, fb = _check_citations("An answer with no citations at all.", 3)
    assert not ok


# ---------- number grounding ----------

def test_check_numbers_fabricated_value_fails():
    evid = "ELECTRA performs comparably while using less than 1/4 of their compute."
    ok, offending = _check_numbers(
        "ELECTRA uses 1/135 of the compute [1].", evid, question=""
    )
    assert not ok
    assert "1/135" in offending[0]

def test_check_numbers_normalized_match_passes():
    evid = "We used warmup_steps = 4000 with 175 billion parameters."
    ok, offending = _check_numbers(
        "It uses 4,000 warmup steps [1] and 175B parameters [1].", evid, question=""
    )
    assert ok

def test_check_numbers_question_numbers_allowed():
    ok, offending = _check_numbers(
        "GPT-3 has more parameters [1].", "the model is larger", question="What about GPT-3?"
    )
    assert ok  # "3" comes from the question, not fabrication

def test_check_numbers_substring_of_larger_number_fails():
    evid = "The model trained for 14000 steps."
    ok, offending = _check_numbers("It trained for 400 steps [1].", evid, question="")
    assert not ok
    assert "400" in offending


def test_check_numbers_sentence_final_period_still_matches():
    evid = "We used warmup_steps = 4000."
    ok, offending = _check_numbers("It uses 4000 warmup steps [1].", evid, question="")
    assert ok


# ---------- support parser (migrated from critique) ----------

def test_parse_support_clean():
    clean, claims = _parse_support("VERDICT: clean")
    assert clean and claims == []

def test_parse_support_unsupported_with_claims():
    raw = "VERDICT: unsupported\nCLAIMS:\n- What optimizer is used?\n- What is the warmup?"
    clean, claims = _parse_support(raw)
    assert not clean
    assert claims == ["What optimizer is used?", "What is the warmup?"]

def test_parse_support_garbage_defaults_clean():
    clean, claims = _parse_support("no verdict here")
    assert clean


# ---------- verify orchestration ----------

def _patch_embeddings(monkeypatch, sim):
    """Make every sub-question/sentence pair have cosine `sim`."""
    def fake_embed(texts):
        v = np.zeros((len(texts), 3), dtype="float32")
        v[:, 0] = 1.0 if sim >= 0.99 else 0.0
        v[:, 1] = 0.0 if sim >= 0.99 else 1.0
        # sub-questions and sentences get identical vectors when sim high,
        # orthogonal when low; caller embeds sub-qs and sentences separately.
        return v
    monkeypatch.setattr(verify_mod, "embed_texts", fake_embed)


def test_verify_fabrication_dispatches_regenerate(monkeypatch):
    monkeypatch.setattr(verify_mod, "chat", lambda *a, **k: "yes")
    _patch_embeddings(monkeypatch, sim=1.0)
    state = {
        "question": "How efficient is ELECTRA?",
        "route": "hard",
        "answer": "ELECTRA uses 1/135 of the compute [1].",
        "retrieved": _evidence(["ELECTRA uses less than 1/4 of their compute."]),
        "sub_questions": ["How efficient is ELECTRA?"],
        "iterations": 1,
        "trace": [],
    }
    r = verify(state)
    assert r["failure_type"] == "fabrication"
    assert r["heal_action"] == "regenerate"
    assert r["iterations"] == 2          # regenerate dispatch pre-increments
    assert not r["critique_clean"]
    assert "1/135" in r["verify_feedback"]


def test_verify_support_failure_dispatches_research(monkeypatch):
    def fake_chat(system, user, **kw):
        if kw.get("label") == "verify-responsiveness":
            return "yes"
        return "VERDICT: unsupported\nCLAIMS:\n- What optimizer does it use?"
    monkeypatch.setattr(verify_mod, "chat", fake_chat)
    _patch_embeddings(monkeypatch, sim=1.0)
    state = {
        "question": "What optimizer?",
        "route": "hard",
        "answer": "It uses Adam [1].",
        "retrieved": _evidence(["Adam optimizer text."]),
        "sub_questions": ["What optimizer?"],
        "iterations": 1,
        "trace": [],
    }
    r = verify(state)
    assert r["failure_type"] == "support"
    assert r["heal_action"] == "research"
    assert r["iterations"] == 1          # search will do the incrementing
    assert r["unsupported_claims"] == ["What optimizer does it use?"]


def test_verify_responsiveness_failure(monkeypatch):
    def fake_chat(system, user, **kw):
        if kw.get("label") == "verify-responsiveness":
            return "no"
        return "VERDICT: clean"
    monkeypatch.setattr(verify_mod, "chat", fake_chat)
    _patch_embeddings(monkeypatch, sim=1.0)
    state = {
        "question": "What optimizer does the Transformer use?",
        "route": "hard",
        "answer": "Self-attention processes sequences [1].",
        "retrieved": _evidence(["Self-attention text about sequences."]),
        "sub_questions": ["What optimizer does the Transformer use?"],
        "iterations": 1,
        "trace": [],
    }
    r = verify(state)
    assert r["failure_type"] == "responsiveness"
    assert r["heal_action"] == "regenerate"


def test_verify_budget_exhausted_honest_exit(monkeypatch):
    monkeypatch.setattr(verify_mod, "chat", lambda *a, **k: "yes")
    _patch_embeddings(monkeypatch, sim=1.0)
    state = {
        "question": "How efficient?",
        "route": "hard",
        "answer": "It uses 1/135 of the compute [1].",
        "retrieved": _evidence(["less than 1/4 of compute."]),
        "sub_questions": ["How efficient?"],
        "iterations": 3,                  # budget already spent
        "trace": [],
    }
    r = verify(state)
    assert r["heal_action"] == "none"
    assert r["verification_warnings"]     # surfaced, not silent
    assert "could not be fully verified" in r["answer"]


def test_verify_medium_skips_stage2_and_caps_at_one_heal(monkeypatch):
    called = {"chat": 0}
    def fake_chat(*a, **k):
        called["chat"] += 1
        return "yes"
    monkeypatch.setattr(verify_mod, "chat", fake_chat)
    state = {
        "question": "What is X?",
        "route": "medium",
        "answer": "X is Y [1].",
        "retrieved": _evidence(["X is Y."]),
        "sub_questions": [],
        "iterations": 1,
        "trace": [],
    }
    r = verify(state)
    assert r["critique_clean"] and r["heal_action"] == "none"
    assert called["chat"] == 0            # medium: Stage 1 only, no LLM

    # medium with a fabricated number at iterations=2: budget (2) exhausted
    state2 = dict(state, answer="X is 999 [1].", iterations=2)
    r2 = verify(state2)
    assert r2["heal_action"] == "none"
    assert r2["verification_warnings"]


def test_verify_clean_passes_everything(monkeypatch):
    def fake_chat(system, user, **kw):
        if kw.get("label") == "verify-responsiveness":
            return "yes"
        return "VERDICT: clean"
    monkeypatch.setattr(verify_mod, "chat", fake_chat)
    _patch_embeddings(monkeypatch, sim=1.0)
    state = {
        "question": "What optimizer?",
        "route": "hard",
        "answer": "Adam with 4000 warmup steps [1].",
        "retrieved": _evidence(["We used the Adam optimizer with warmup_steps = 4000."]),
        "sub_questions": ["What optimizer?"],
        "iterations": 1,
        "trace": [],
    }
    r = verify(state)
    assert r["critique_clean"] is True
    assert r["failure_type"] == ""
    assert r["heal_action"] == "none"
    assert r["verification_warnings"] == []
