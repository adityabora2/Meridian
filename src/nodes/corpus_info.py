from __future__ import annotations

try:
    from src.ingest import load_index
    from src.state import RAGState
except ImportError:
    from ingest import load_index  # type: ignore
    from state import RAGState  # type: ignore


def corpus_info(state: RAGState) -> RAGState:
    trace = list(state.get("trace", []))
    try:
        _, metadata = load_index()
    except Exception:
        trace.append("corpus_info → no index")
        return {
            "answer": (
                "No documents have been indexed yet. Add PDFs to data/documents/ "
                "and run `python -m src.ingest`."
            ),
            "citations": [],
            "trace": trace,
        }

    seen: dict[str, str] = {}
    for chunk in metadata:
        if chunk.document_name not in seen:
            seen[chunk.document_name] = chunk.document_title

    lines = [
        f"- {name} — {title}" if title else f"- {name}"
        for name, title in seen.items()
    ]
    answer = (
        f"There are {len(seen)} document(s) indexed:\n" + "\n".join(lines)
    )
    trace.append(f"corpus_info → {len(seen)} document(s)")
    return {"answer": answer, "citations": [], "trace": trace}
