import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import _score_document_match, match_document


def test_score_document_match_counts_token_overlap():
    query_tokens = {"how", "does", "bert", "handle", "positional", "encoding"}
    doc_tokens = {"bert", "pretraining", "deep", "bidirectional", "transformers"}
    score = _score_document_match(query_tokens, doc_tokens)
    assert score >= 1  # "bert" overlaps


def test_match_document_returns_clear_winner(monkeypatch):
    fake_docs = {
        "bert.pdf": "bert pretraining deep bidirectional transformers language understanding",
        "t5.pdf": "exploring limits transfer learning unified text to text transformer",
    }

    import src.ingest as ingest_module

    monkeypatch.setattr(ingest_module, "_document_match_corpus", lambda: fake_docs)

    result = match_document("How does BERT handle positional encoding?")
    assert result == "bert.pdf"


def test_match_document_returns_none_when_ambiguous(monkeypatch):
    fake_docs = {
        "doc_a.pdf": "machine learning transformer attention model",
        "doc_b.pdf": "machine learning transformer attention model",
    }

    import src.ingest as ingest_module

    monkeypatch.setattr(ingest_module, "_document_match_corpus", lambda: fake_docs)

    result = match_document("Tell me about the machine learning transformer model")
    assert result is None


def test_match_document_returns_none_for_generic_cross_document_question(monkeypatch):
    fake_docs = {
        "bert.pdf": "bert pretraining deep bidirectional transformers",
        "t5.pdf": "exploring limits transfer learning text to text transformer",
    }

    import src.ingest as ingest_module

    monkeypatch.setattr(ingest_module, "_document_match_corpus", lambda: fake_docs)

    result = match_document("Compare how different papers approach pretraining objectives")
    assert result is None


if __name__ == "__main__":
    test_score_document_match_counts_token_overlap()
    print("=== test_score_document_match_counts_token_overlap PASSED ===")
