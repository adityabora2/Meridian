import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nodes.critique import _parse_critique


def test_parses_clean_verdict():
    raw = "VERDICT: clean"
    clean, claims = _parse_critique(raw)
    assert clean is True
    assert claims == []


def test_parses_unsupported_with_claims():
    raw = (
        "VERDICT: unsupported\n"
        "CLAIMS:\n"
        "- What dataset did Self-RAG use for evaluation?\n"
        "- What is the exact reflection token vocabulary size?\n"
    )
    clean, claims = _parse_critique(raw)
    assert clean is False
    assert claims == [
        "What dataset did Self-RAG use for evaluation?",
        "What is the exact reflection token vocabulary size?",
    ]


def test_unparseable_response_falls_back_to_clean():
    raw = "I think the answer looks fine overall, no clear verdict here."
    clean, claims = _parse_critique(raw)
    assert clean is True
    assert claims == []


def test_empty_response_falls_back_to_clean():
    clean, claims = _parse_critique("")
    assert clean is True
    assert claims == []


def test_unsupported_with_no_claim_lines_falls_back_to_clean():
    raw = "VERDICT: unsupported\nCLAIMS:\n"
    clean, claims = _parse_critique(raw)
    assert clean is True
    assert claims == []


if __name__ == "__main__":
    test_parses_clean_verdict()
    test_parses_unsupported_with_claims()
    test_unparseable_response_falls_back_to_clean()
    test_empty_response_falls_back_to_clean()
    test_unsupported_with_no_claim_lines_falls_back_to_clean()
    print("=== all critique parsing tests PASSED ===")
