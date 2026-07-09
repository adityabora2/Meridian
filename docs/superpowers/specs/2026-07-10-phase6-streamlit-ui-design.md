# Phase 6 — Streamlit UI (`app.py`)

## Context

Phases 0-5 built and verified the full pipeline: ingestion (Phase 2), the router and
Mode 1/2 nodes (Phase 3), the Mode 3 nodes (Phase 4), and the compiled LangGraph
`StateGraph` wiring all of it together (Phase 5 — `build_graph()` in `src/graph.py`,
returning a `CompiledStateGraph` with a plain `invoke(dict) -> dict` interface,
already live-verified across all three modes). Phase 6 is the last piece needed to
actually demo the project: a single-page Streamlit UI that calls `build_graph()`
directly (no API layer, per PLAN.md) and presents the answer, which mode fired, and
citations.

## Layout

Single file `app.py` at the repo root. Top to bottom:

1. **Title and one-line description** of the project.
2. **Question input** — a `st.selectbox` of three preset example questions (reusing
   the exact three questions already live-verified in Phase 5's dry-runs, one per
   mode) plus a `st.text_input` for free-text entry, defaulting to the selectbox's
   current value so either path feeds the same variable.
3. **Run button.**
4. **On click:** `st.spinner(...)` wraps a single
   `graph.invoke({"question": question, "trace": []})` call inside a `try/except`
   block. On exception, `st.error(...)` shows the exception message — no raw
   Streamlit traceback.
5. **On success — the headline:** a visually prominent mode badge built from
   `state["mode_label"]` (e.g. `"Mode 3 - Multi-Hop + Self-Critique"`), colored
   distinctly per mode (`st.success`/`st.info`/`st.warning`, one per mode), followed
   by the answer text, followed by citations rendered as a list of
   `document_name` + `page_number` pairs from `state["citations"]`.
6. **Collapsed `st.expander("How this answer was produced")`** below the headline,
   containing: the iteration count (`state["iterations"]`, only meaningful for Mode
   3), the `critique_clean` verdict if present in state, and the full `trace` list
   (`state["trace"]`) rendered as an ordered list of strings.

## Interaction and state details

- `build_graph()` is called once at module load. Streamlit reruns the whole script
  on every interaction, but graph compilation is cheap and stateless (confirmed by
  the Phase 5 final review: `build_graph()` is a factory, not a module global holding
  mutable state), so no caching/session-state complexity is needed for a demo of this
  size.
- The three preset example questions (exact strings, reused verbatim from Phase 5's
  live-verified dry-runs so they are known to route correctly):
  - Mode 1: `"What is a vector database?"`
  - Mode 2: `"According to the BERT paper, what pretraining tasks does BERT use?"`
  - Mode 3: `"Compare how BERT and the Transformer paper each handle positional information, and explain any tradeoffs."`
- `trace: []` is always passed explicitly in the initial state dict (per the Phase 5
  reviewer's recommendation — nodes tolerate a missing `trace` via
  `state.get("trace", [])`, but passing it explicitly matches the exact path already
  verified live).

## Hard constraint: no AI-generated-looking text artifacts

Every user-visible string in `app.py` (page title, captions, labels, button text,
badge text, error messages, expander title, any other on-screen text) must:
- Contain **no emoji or pictographic characters** anywhere.
- Use **regular hyphens (`-`), never em dashes (`—`) or en dashes (`–`)**, in any
  on-screen string — including the mode label display, since `config.MODE_LABELS`
  currently contains a middle-dot character (`"Mode 1 · No Retrieval"`) which is
  acceptable (not an em dash, not an emoji), but if `app.py` composes any new string
  around it, that composition must not introduce a dash character of the em/en
  variety.
- This constraint applies to **user-visible strings only** — it does not extend to
  Python code comments in `app.py` (of which there should be few to none, per this
  repo's existing no-comments-by-default convention).
- This is a deliberate stylistic requirement (avoiding a look that reads as
  AI-generated boilerplate), not a technical constraint — call it out explicitly in
  the implementation plan so it survives as a checkable requirement, not an implicit
  assumption.

## No new dependencies

`streamlit==1.41.1` is already in `requirements.txt` (installed and smoke-tested in
Phase 1). No other new packages needed.

## Testing approach

Per the guidance to verify UI changes by actually driving them in a browser: after
implementation, run `streamlit run app.py`, then exercise all three preset questions
through the running app and confirm by direct observation (not just code reading):
- The mode badge displays correctly and distinctly for each of the three modes.
- The answer and citations render correctly for Mode 2 and Mode 3 (Mode 1 has no
  citations by design).
- The expander shows the trace and, for Mode 3, an iteration count consistent with
  what was already observed in Phase 5's live dry-run (up to 3).
- No emoji or em/en dash appears anywhere in the rendered page for any of the three
  runs.
- Free-text entry (typing a question not in the preset list) also works end-to-end.
- Trigger the error path once (e.g. temporarily point `GROQ_API_KEY` at an invalid
  value, or simulate by another means) to confirm `st.error(...)` shows a clean
  message rather than a raw traceback, then restore the valid key.

## Out of scope for Phase 6

- Any change to `src/graph.py` or any node file.
- The 30-question test set (Phase 7).
- Streaming/live step-by-step UI updates (would require `graph.stream()` instead of
  `graph.invoke()` — explicitly deferred, not requested by PLAN.md).
- Session history, multi-turn conversation, or any state persistence beyond a single
  run.
