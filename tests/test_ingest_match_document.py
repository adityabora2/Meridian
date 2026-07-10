import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import _score_document_match, _tokenize, match_document


def test_tokenize_dehyphenates_model_names():
    # "GPT-2" must yield "gpt2" (matching a filename stem) in addition to the
    # split "gpt"/"2" tokens, so hyphenated model names match their document.
    tokens = _tokenize("How does GPT-2 work?")
    assert "gpt2" in tokens
    assert "gpt" in tokens


def test_match_document_distinguishes_hyphenated_siblings(monkeypatch):
    # gpt2/gpt3 have near-identical titles; only the de-hyphenated stem match
    # ("gpt2" -> gpt2.pdf) can tell them apart. Without de-hyphenation both
    # tie and fall back to None (a recall miss this test guards against).
    fake_docs = {
        "gpt2.pdf": "language models are unsupervised multitask learners gpt2",
        "gpt3.pdf": "language models are few shot learners gpt3",
    }

    import src.ingest as ingest_module

    monkeypatch.setattr(ingest_module, "_document_match_corpus", lambda: fake_docs)

    assert match_document("How does GPT-2 do unsupervised multitask learning?") == "gpt2.pdf"
    assert match_document("How does GPT-3 do few-shot learning?") == "gpt3.pdf"


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


def test_implicated_documents_two_named_docs(monkeypatch):
    from src import ingest
    monkeypatch.setattr(
        ingest, "_document_match_corpus",
        lambda: {
            "bert.pdf": "BERT: Pre-training of Deep Bidirectional Transformers bert",
            "roberta.pdf": "RoBERTa: A Robustly Optimized BERT Pretraining Approach roberta",
            "t5.pdf": "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer t5",
        },
    )
    got = ingest.implicated_documents(
        "How does RoBERTa's pretraining differ from BERT's?"
    )
    assert got == ["bert.pdf", "roberta.pdf"]


def test_implicated_documents_stem_only_ignores_title_words(monkeypatch):
    from src import ingest
    monkeypatch.setattr(
        ingest, "_document_match_corpus",
        lambda: {
            "attention_is_all_you_need.pdf": "Attention Is All You Need attention_is_all_you_need",
            "t5.pdf": "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer t5",
        },
    )
    # "Transformer" appears in t5's TITLE but matches no filename stem: nothing implicated.
    got = ingest.implicated_documents(
        "What optimizer does the original Transformer use?"
    )
    assert got == []


def test_implicated_documents_dehyphenated_model_name(monkeypatch):
    from src import ingest
    monkeypatch.setattr(
        ingest, "_document_match_corpus",
        lambda: {
            "gpt2.pdf": "Language Models are Unsupervised Multitask Learners gpt2",
            "gpt3.pdf": "Language Models are Few-Shot Learners gpt3",
        },
    )
    got = ingest.implicated_documents("Trace the evolution from GPT-2 to GPT-3")
    assert got == ["gpt2.pdf", "gpt3.pdf"]


def test_implicated_documents_empty_corpus(monkeypatch):
    from src import ingest
    monkeypatch.setattr(ingest, "_document_match_corpus", lambda: {})
    assert ingest.implicated_documents("anything") == []


if __name__ == "__main__":
    test_score_document_match_counts_token_overlap()
    test_tokenize_dehyphenates_model_names()
    print("=== non-monkeypatch match_document tests PASSED ===")
