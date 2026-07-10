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


def test_parse_label_recognizes_meta():
    assert _parse_label("meta") == "meta"
    assert _parse_label("  META  ") == "meta"
    assert _parse_label("This looks like a meta question.") == "meta"


def test_parse_label_unparseable_still_none_not_meta():
    # Regression: garbage must fall back via None (-> medium in route_question),
    # never accidentally resolve to meta.
    assert _parse_label("banana") is None


if __name__ == "__main__":
    test_parse_label_exact_match()
    test_parse_label_strips_whitespace_and_case()
    test_parse_label_extracts_from_a_sentence()
    test_parse_label_returns_none_when_unparseable()
    print("=== non-monkeypatch router tests PASSED ===")
