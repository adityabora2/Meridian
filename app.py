from __future__ import annotations

import streamlit as st

from src.graph import build_graph
from src.logging_config import setup_logging

setup_logging()

st.set_page_config(page_title="Adaptive RAG", layout="centered")


@st.cache_resource
def _graph():
    return build_graph()


MODE_CAPTION = {
    "easy": "Answered directly (no retrieval needed)",
    "medium": "Answered via single-hop retrieval",
    "hard": "Answered via multi-hop retrieval with self-critique",
}


def _render_answer(result: dict) -> None:
    """Render one assistant turn: a small mode caption, the answer, citations,
    and a collapsible detail panel. The mode is chosen by the router on its own;
    this only reports what it decided."""
    route = result.get("route", "medium")
    st.caption(MODE_CAPTION.get(route, "Answered"))

    st.write(result.get("answer", ""))

    citations = result.get("citations", [])
    if citations:
        lines = [
            f"- {c.get('document_name')}, page {c.get('page_number')}"
            for c in citations
        ]
        st.markdown("**Sources**\n" + "\n".join(lines))

    with st.expander("How this answer was produced"):
        st.write(f"Mode: {result.get('mode_label', route)}")
        if route == "hard":
            st.write(f"Iterations: {result.get('iterations', 0)}")
            st.write(f"Critique clean: {result.get('critique_clean')}")
        trace = result.get("trace", [])
        if trace:
            st.write("Trace:")
            for i, step in enumerate(trace, start=1):
                st.write(f"{i}. {step}")


st.title("Adaptive RAG")
st.caption(
    "Ask a question about your documents. The system decides on its own how "
    "much retrieval and verification each question needs, then answers."
)

if "history" not in st.session_state:
    st.session_state.history = []  # list of {"question": str, "result": dict}

# Replay the conversation so far.
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        _render_answer(turn["result"])

question = st.chat_input("Ask a question")
if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = _graph().invoke({"question": question, "trace": []})
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")
                result = None
        if result is not None:
            _render_answer(result)
            st.session_state.history.append({"question": question, "result": result})
