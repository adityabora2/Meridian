"""Offline tests for the verification gate. chat and embed_texts are
monkeypatched; no Ollama or network needed.

Checks are now pure `(VerifyContext) -> Failure | None` functions in a registry,
so each is tested directly, and the registry's own invariants (route
applicability, cheap-before-expensive ordering, a heal action per failure type)
are tested as data rather than inferred from the node's branching.
"""
import numpy as np

from meridian import config
from meridian.nodes import verify as verify_mod
from meridian.nodes.verify import (
    CHECKS,
    HEAL_ACTIONS,
    VerifyContext,
    _canon_number,
    _check_citations,
    _check_numbers,
    _parse_support,
    run_checks,
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


def _ctx(answer, evidence_texts=(), *, question="", route="hard", sub_questions=()):
    return VerifyContext(
        question=question,
        answer=answer,
        route=route,
        sub_questions=tuple(sub_questions),
        pool=_evidence(list(evidence_texts)),
    )


# ---------- number canonicalization ----------

def test_canon_number_folds_commas_and_magnitudes():
    assert _canon_number("4,000") == "4000"
    assert _canon_number("175B") == "175billion"
    assert _canon_number("175 billion") == "175billion"
    assert _canon_number("93.3%") == "93.3%"
    assert _canon_number("1/4") == "1/4"


# ---------- citation check ----------

def test_check_citations_ok():
    assert _check_citations(_ctx("Adam is used [1] with warmup [2].", ["a", "b", "c"])) is None


def test_check_citations_accepts_a_range():
    # REGRESSION: this used to fail as "no [n] citations" and send a correctly
    # grounded answer back through the heal loop for a formatting reason.
    assert _check_citations(_ctx("All of it is supported [1-3].", ["a", "b", "c"])) is None


def test_check_citations_out_of_range():
    failure = _check_citations(_ctx("Adam is used [7].", ["a", "b", "c"]))
    assert failure is not None
    assert failure.failure_type == "citations"
    assert "7" in failure.feedback


def test_check_citations_malformed_marker():
    failure = _check_citations(_ctx("PaLM outperforms [n14] and [n].", ["a", "b", "c"]))
    assert failure is not None
    assert "malformed" in failure.feedback


def test_check_citations_none_present():
    failure = _check_citations(_ctx("An answer with no citations at all.", ["a", "b", "c"]))
    assert failure is not None
    assert "no [n] citations" in failure.feedback


# ---------- number grounding ----------

def test_check_numbers_fabricated_value_fails():
    evid = "ELECTRA performs comparably while using less than 1/4 of their compute."
    failure = _check_numbers(_ctx("ELECTRA uses 1/135 of the compute [1].", [evid]))
    assert failure is not None
    assert failure.failure_type == "fabrication"
    assert "1/135" in failure.feedback


def test_check_numbers_normalized_match_passes():
    evid = "We used warmup_steps = 4000 with 175 billion parameters."
    assert _check_numbers(
        _ctx("It uses 4,000 warmup steps [1] and 175B parameters [1].", [evid])
    ) is None


def test_check_numbers_question_numbers_allowed():
    # "3" comes from the question, not fabrication.
    assert _check_numbers(
        _ctx("GPT-3 has more parameters [1].", ["the model is larger"],
             question="What about GPT-3?")
    ) is None


def test_check_numbers_substring_of_larger_number_fails():
    failure = _check_numbers(
        _ctx("It trained for 400 steps [1].", ["The model trained for 14000 steps."])
    )
    assert failure is not None
    assert "400" in failure.feedback


def test_check_numbers_sentence_final_period_still_matches():
    assert _check_numbers(
        _ctx("It uses 4000 warmup steps [1].", ["We used warmup_steps = 4000."])
    ) is None


def test_check_numbers_ignores_indices_inside_a_citation_range():
    # REGRESSION: "[1-7,9]" must be stripped before number extraction, or 1, 7
    # and 9 are read as claimed figures absent from the evidence.
    assert _check_numbers(
        _ctx("The approach is well supported [1-7,9].", ["some evidence with no digits"])
    ) is None


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


# ---------- the registry itself ----------

def test_every_check_has_a_heal_action():
    assert {c.name for c in CHECKS} == set(HEAL_ACTIONS)


def test_heal_actions_are_valid_graph_destinations():
    assert set(HEAL_ACTIONS.values()) <= {"regenerate", "research"}


def test_deterministic_checks_run_before_llm_checks():
    names = [c.name for c in CHECKS]
    first_llm = min(i for i, c in enumerate(CHECKS) if c.uses_llm)
    assert not any(c.uses_llm for c in CHECKS[:first_llm]), names


def test_medium_route_runs_no_llm_checks():
    applicable = [c for c in CHECKS if c.applies_to(config.ROUTE_MEDIUM)]
    assert applicable, "medium must still be verified"
    assert not any(c.uses_llm for c in applicable)


def test_hard_route_runs_every_check():
    assert all(c.applies_to(config.ROUTE_HARD) for c in CHECKS)


def test_run_checks_stops_at_the_first_failure(monkeypatch):
    # A malformed citation must not cost an LLM call.
    def _boom(*a, **k):
        raise AssertionError("no LLM check should run after a deterministic failure")

    monkeypatch.setattr(verify_mod, "chat", _boom)
    failure = run_checks(_ctx("Answer with [n].", ["evidence"], route="hard"))
    assert failure.failure_type == "citations"


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
    }
    r = verify(state)
    assert r["failure_type"] == "fabrication"
    assert r["heal_action"] == "regenerate"
    assert not r["critique_clean"]
    assert "1/135" in r["verify_feedback"]


def test_verify_never_writes_iterations(monkeypatch):
    # generate owns the counter; verify only reads it for the budget check.
    monkeypatch.setattr(verify_mod, "chat", lambda *a, **k: "yes")
    _patch_embeddings(monkeypatch, sim=1.0)
    state = {
        "question": "How efficient is ELECTRA?",
        "route": "hard",
        "answer": "ELECTRA uses 1/135 of the compute [1].",
        "retrieved": _evidence(["ELECTRA uses less than 1/4 of their compute."]),
        "sub_questions": ["How efficient is ELECTRA?"],
        "iterations": 1,
    }
    assert "iterations" not in verify(state)


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
    }
    r = verify(state)
    assert r["failure_type"] == "support"
    assert r["heal_action"] == "research"
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
    }
    r = verify(state)
    assert r["failure_type"] == "responsiveness"
    assert r["heal_action"] == "regenerate"


def _patch_orthogonal_embeddings(monkeypatch):
    """Sub-questions and answer sentences get orthogonal vectors, so every
    sub-question reads as uncovered. verify embeds sub-questions first."""
    calls = {"n": 0}

    def fake_embed(texts):
        v = np.zeros((len(texts), 2), dtype="float32")
        v[:, calls["n"] % 2] = 1.0
        calls["n"] += 1
        return v

    monkeypatch.setattr(verify_mod, "embed_texts", fake_embed)


def test_verify_coverage_failure_is_hard_only(monkeypatch):
    monkeypatch.setattr(verify_mod, "chat", lambda *a, **k: "yes")
    _patch_orthogonal_embeddings(monkeypatch)   # nothing covers anything
    state = {
        "question": "Compare A and B.",
        "route": "hard",
        "answer": "This is a long enough sentence about something unrelated [1].",
        "retrieved": _evidence(["evidence text"]),
        "sub_questions": ["What is A?", "What is B?"],
        "iterations": 1,
    }
    assert verify(state)["failure_type"] == "coverage"

    # Same state on the medium route: coverage does not apply.
    medium = dict(state, route="medium")
    assert verify(medium)["critique_clean"] is True


def test_verify_budget_exhausted_honest_exit(monkeypatch):
    monkeypatch.setattr(verify_mod, "chat", lambda *a, **k: "yes")
    _patch_embeddings(monkeypatch, sim=1.0)
    state = {
        "question": "How efficient?",
        "route": "hard",
        "answer": "It uses 1/135 of the compute [1].",
        "retrieved": _evidence(["less than 1/4 of compute."]),
        "sub_questions": ["How efficient?"],
        "iterations": config.MAX_ITERATIONS,   # budget already spent
    }
    r = verify(state)
    assert r["heal_action"] == "none"
    assert r["verification_warnings"]     # surfaced, not silent
    assert "could not be fully verified" in r["answer"]


def test_verify_medium_skips_llm_checks_and_caps_at_one_heal(monkeypatch):
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
    }
    r = verify(state)
    assert r["critique_clean"] and r["heal_action"] == "none"
    assert called["chat"] == 0            # medium: deterministic checks only

    # medium with a fabricated number at its cap: budget exhausted.
    state2 = dict(state, answer="X is 999 [1].", iterations=config.MEDIUM_ITERATION_CAP)
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
    }
    r = verify(state)
    assert r["critique_clean"] is True
    assert r["failure_type"] == ""
    assert r["heal_action"] == "none"
    assert r["verification_warnings"] == []


def test_verify_short_circuits_with_no_evidence(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("no check should run without evidence")

    monkeypatch.setattr(verify_mod, "chat", _boom)
    r = verify({"question": "q", "route": "hard", "answer": "an answer", "retrieved": []})
    assert r["critique_clean"] is True
    assert r["heal_action"] == "none"


def test_check_citations_truncates_a_long_out_of_range_list():
    # "[1-500]" parses cleanly but is almost all out of range; the feedback goes
    # straight back into the regenerate prompt, so it must not list 488 indices.
    failure = _check_citations(_ctx("Claim [1-500].", ["a", "b", "c"]))
    assert failure is not None
    assert "more)" in failure.feedback
    assert len(failure.feedback) < 200


def test_unlisted_route_still_runs_the_deterministic_checks():
    # Only medium and hard reach verify today, but a check scoped to a listed
    # set would silently stop running on any other value -- verification must
    # not fail open on an unexpected route.
    failure = run_checks(_ctx("No citations here.", ["evidence"], route="something-new"))
    assert failure is not None
    assert failure.failure_type == "citations"
