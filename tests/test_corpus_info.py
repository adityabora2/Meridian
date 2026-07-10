import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nodes.corpus_info import corpus_info


class _FakeChunk:
    def __init__(self, name, title):
        self.document_name = name
        self.document_title = title


def test_corpus_info_lists_distinct_documents(monkeypatch):
    import src.nodes.corpus_info as mod

    chunks = [
        _FakeChunk("bert.pdf", "BERT: Pre-training"),
        _FakeChunk("bert.pdf", "BERT: Pre-training"),   # duplicate doc, one entry
        _FakeChunk("t5.pdf", "Exploring the Limits"),
    ]
    monkeypatch.setattr(mod, "load_index", lambda: (None, chunks))

    result = corpus_info({"question": "what documents are loaded?", "trace": []})
    answer = result["answer"]
    assert "bert.pdf" in answer
    assert "t5.pdf" in answer
    assert "BERT: Pre-training" in answer
    assert "2" in answer  # a count of 2 distinct documents
    assert result["citations"] == []
    assert answer.count("bert.pdf") == 1  # deduplicated


def test_corpus_info_handles_missing_index(monkeypatch):
    import src.nodes.corpus_info as mod

    def _raise():
        raise FileNotFoundError("no index")

    monkeypatch.setattr(mod, "load_index", _raise)

    result = corpus_info({"question": "what documents are loaded?", "trace": []})
    assert "no documents" in result["answer"].lower() or "not been built" in result["answer"].lower()
    assert result["citations"] == []


if __name__ == "__main__":
    print("run via pytest (uses monkeypatch)")
