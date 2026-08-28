from __future__ import annotations

from meridian.ingest import load_index
from meridian.logging_config import get_logger
from meridian.state import RAGState

log = get_logger("corpus_info")


def corpus_info(state: RAGState) -> RAGState:
    try:
        _, metadata = load_index()
    except Exception:
        return {
            "answer": (
                "No documents have been indexed yet. Add PDFs to data/documents/ "
                "and run `python -m meridian.ingest`."
            ),
            "citations": [],
            "trace": ["corpus_info → no index"],
        }

    seen: dict[str, str] = {}
    for chunk in metadata:
        if chunk.document_name not in seen:
            seen[chunk.document_name] = chunk.document_title

    lines = [
        f"- {name}: {title}" if title else f"- {name}"
        for name, title in seen.items()
    ]
    answer = f"There are {len(seen)} document(s) indexed:\n" + "\n".join(lines)
    log.info("corpus listing -> %d document(s)", len(seen))
    return {
        "answer": answer,
        "citations": [],
        "trace": [f"corpus_info → {len(seen)} document(s)"],
    }
