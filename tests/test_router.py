import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nodes.router import _parse_label, route_question


def test_parse_label_exact_match():
    assert _parse_label("easy") == "easy"
    assert _parse_label("medium") == "medium"
    assert _parse_label("hard") == "hard"


def test_parse_label_strips_whitespace_and_case():
    assert _parse_label("  EASY  ") == "easy"
    assert _parse_label("Hard\n") == "hard"


def test_parse_label_extracts_from_a_sentence():
    # The model sometimes wraps the label in prose despite instructions.
    assert _parse_label("This question is medium complexity.") == "medium"
    assert _parse_label("I would classify this as hard.") == "hard"


def test_parse_label_returns_none_when_unparseable():
    assert _parse_label("") is None
    assert _parse_label("banana") is None
    assert _parse_label("42") is None


def test_route_question_falls_back_to_medium_on_unparseable(monkeypatch):
    import src.nodes.router as router_module

    # Simulate the LLM returning garbage the parser can't resolve.
    monkeypatch.setattr(router_module, "chat", lambda *a, **k: "???")

    result = route_question({"question": "anything", "trace": []})
    assert result["route"] == "medium"
    assert result["mode_label"] == router_module.config.MODE_LABELS["medium"]
    assert result["iterations"] == 0
    assert result["trace"] == ["router → medium"]


def test_route_question_uses_parsed_label(monkeypatch):
    import src.nodes.router as router_module

    monkeypatch.setattr(router_module, "chat", lambda *a, **k: "hard")

    result = route_question({"question": "compare X and Y", "trace": []})
    assert result["route"] == "hard"
    assert result["mode_label"] == router_module.config.MODE_LABELS["hard"]
    assert result["iterations"] == 0


def test_route_question_resets_iterations_and_preserves_prior_trace(monkeypatch):
    import src.nodes.router as router_module

    monkeypatch.setattr(router_module, "chat", lambda *a, **k: "easy")

    # A stale iterations value and prior trace should be reset/preserved
    # correctly: iterations back to 0, trace appended (not replaced).
    result = route_question(
        {"question": "q", "iterations": 5, "trace": ["earlier step"]}
    )
    assert result["iterations"] == 0
    assert result["trace"] == ["earlier step", "router → easy"]


# --- meta detection is deterministic (is_meta_question), not an LLM label ---


def test_is_meta_question_detects_corpus_listing(monkeypatch):
    import src.nodes.router as router_module
    # No specific document is named in these, so match_document returns None.
    monkeypatch.setattr(router_module, "match_document", lambda q: None)
    from src.nodes.router import is_meta_question

    assert is_meta_question("what documents are loaded?")
    assert is_meta_question("which files can I ask about?")
    assert is_meta_question("how many documents are indexed?")
    assert is_meta_question("list all the papers")
    assert is_meta_question("what documents are ingested?")
    assert is_meta_question("explain the documents")


def test_is_meta_question_rejects_content_questions(monkeypatch):
    import src.nodes.router as router_module
    # These name/imply a specific document OR aren't corpus-listing questions.
    monkeypatch.setattr(router_module, "match_document", lambda q: None)
    from src.nodes.router import is_meta_question

    assert not is_meta_question("Hi")
    assert not is_meta_question("who are the authors in xlnet")
    assert not is_meta_question("what all authors are there in xlnet")
    assert not is_meta_question("what is machine learning")
    assert not is_meta_question("explain bert")


def test_is_meta_question_named_document_is_content_not_meta(monkeypatch):
    import src.nodes.router as router_module
    # "explain the bert document" pattern-matches a meta frame, but names a
    # specific document -> it's a content question, not a corpus listing.
    monkeypatch.setattr(router_module, "match_document", lambda q: "bert.pdf")
    from src.nodes.router import is_meta_question

    assert not is_meta_question("explain the bert document")


def test_route_question_meta_short_circuits_before_llm(monkeypatch):
    import src.nodes.router as router_module

    def _boom(*a, **k):
        raise AssertionError("LLM chat() must not be called for a meta question")

    monkeypatch.setattr(router_module, "chat", _boom)
    monkeypatch.setattr(router_module, "match_document", lambda q: None)

    result = router_module.route_question({"question": "what documents are loaded", "trace": []})
    assert result["route"] == "meta"
    assert result["trace"] == ["router → meta"]


def test_route_question_upgrades_easy_to_medium_when_document_named(monkeypatch):
    import src.nodes.router as router_module
    monkeypatch.setattr(router_module, "chat", lambda *a, **k: "easy")
    monkeypatch.setattr(router_module, "match_document", lambda q: "bert.pdf")

    result = router_module.route_question({"question": "explain bert", "trace": []})
    assert result["route"] == "medium"
    assert any("upgrad" in t for t in result["trace"])


def test_route_question_keeps_easy_when_no_document_named(monkeypatch):
    import src.nodes.router as router_module
    monkeypatch.setattr(router_module, "chat", lambda *a, **k: "easy")
    monkeypatch.setattr(router_module, "match_document", lambda q: None)

    result = router_module.route_question({"question": "what is machine learning", "trace": []})
    assert result["route"] == "easy"


if __name__ == "__main__":
    test_parse_label_exact_match()
    test_parse_label_strips_whitespace_and_case()
    test_parse_label_extracts_from_a_sentence()
    test_parse_label_returns_none_when_unparseable()
    print("=== non-monkeypatch router tests PASSED ===")
