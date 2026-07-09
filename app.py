from __future__ import annotations

import streamlit as st

from src.graph import build_graph

st.set_page_config(page_title="Adaptive RAG", layout="centered")

PRESET_QUESTIONS = {
    "Mode 1 example - general knowledge": "What is a vector database?",
    "Mode 2 example - single document lookup": (
        "According to the BERT paper, what pretraining tasks does BERT use?"
    ),
    "Mode 3 example - multi-hop comparison": (
        "Compare how BERT and the Transformer paper each handle positional "
        "information, and explain any tradeoffs."
    ),
}

MODE_DISPLAY = {
    "easy": st.success,
    "medium": st.info,
    "hard": st.warning,
}


graph = build_graph()

st.title("Adaptive RAG")
st.caption(
    "Ask a question. The router decides how much retrieval and verification "
    "the question needs, then answers accordingly."
)

preset_label = st.selectbox(
    "Try an example question, or type your own below",
    options=list(PRESET_QUESTIONS.keys()),
)
question = st.text_input(
    "Your question",
    value=PRESET_QUESTIONS[preset_label],
)

if st.button("Run"):
    spinner_text = (
        "Running the adaptive RAG pipeline... "
        "(this may take longer if the API is briefly rate-limited)"
    )
    with st.spinner(spinner_text):
        try:
            result = graph.invoke({"question": question, "trace": []})
        except Exception as exc:
            st.error(f"Something went wrong: {exc}")
        else:
            route = result.get("route", "medium")
            mode_label = result.get("mode_label", "")
            display_fn = MODE_DISPLAY.get(route, st.info)
            display_fn(mode_label)

            st.write(result.get("answer", ""))

            citations = result.get("citations", [])
            if citations:
                st.subheader("Citations")
                for c in citations:
                    st.write(
                        f"[{c.get('marker')}] {c.get('document_name')}, "
                        f"page {c.get('page_number')}"
                    )

            with st.expander("How this answer was produced"):
                if route == "hard":
                    st.write(f"Iterations: {result.get('iterations', 0)}")
                    st.write(f"Critique clean: {result.get('critique_clean')}")
                trace = result.get("trace", [])
                if trace:
                    st.write("Trace:")
                    for i, step in enumerate(trace, start=1):
                        st.write(f"{i}. {step}")
