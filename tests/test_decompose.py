import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nodes.decompose import _parse_sub_questions


def test_parses_multiple_lines():
    raw = "How does Adaptive-RAG route questions?\nHow does Self-RAG use reflection tokens?"
    result = _parse_sub_questions(raw, fallback="orig question")
    assert result == [
        "How does Adaptive-RAG route questions?",
        "How does Self-RAG use reflection tokens?",
    ]


def test_strips_blank_lines():
    raw = "\n  First question?  \n\n\nSecond question?\n\n"
    result = _parse_sub_questions(raw, fallback="orig question")
    assert result == ["First question?", "Second question?"]


def test_caps_at_three():
    raw = "Q1?\nQ2?\nQ3?\nQ4?\nQ5?"
    result = _parse_sub_questions(raw, fallback="orig question")
    assert result == ["Q1?", "Q2?", "Q3?"]


def test_empty_response_falls_back_to_original_question():
    result = _parse_sub_questions("", fallback="orig question")
    assert result == ["orig question"]


def test_whitespace_only_response_falls_back():
    result = _parse_sub_questions("   \n\n   ", fallback="orig question")
    assert result == ["orig question"]


if __name__ == "__main__":
    test_parses_multiple_lines()
    test_strips_blank_lines()
    test_caps_at_three()
    test_empty_response_falls_back_to_original_question()
    test_whitespace_only_response_falls_back()
    print("=== all decompose parsing tests PASSED ===")
