# Phase 6 — Streamlit UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `app.py`, a single-page Streamlit UI that calls the compiled
LangGraph `build_graph()` directly (no API layer) and presents the answer, mode
badge, citations, and a collapsible trace/iteration detail view, per the approved
spec at `docs/superpowers/specs/2026-07-10-phase6-streamlit-ui-design.md`.

**Architecture:** One new file, `app.py`, at the repo root. Calls `build_graph()`
once at module load, takes a question via a preset dropdown or free-text input, runs
it through `graph.invoke({"question": ..., "trace": []})` inside a spinner and a
try/except, and renders the result: a colored mode badge, the answer, citations, and
a collapsed expander with the trace/iteration/critique detail.

**Tech Stack:** `streamlit==1.41.1` (already installed, no new dependencies).
`src.graph.build_graph()` (Phase 5, already built and reviewed) is the only
project-internal import needed.

## Global Constraints

- No changes to any file under `src/` — `app.py` only calls `build_graph()` and reads
  the returned state dict; it does not modify any node, `graph.py`, or `config.py`.
- **No emoji or pictographic characters anywhere in user-visible strings** (titles,
  captions, labels, button text, badge text, error messages, expander title, any
  other on-screen text).
- **No em dashes (`—`) or en dashes (`–`) in any user-visible string** — use regular
  hyphens (`-`) only. This includes any new string `app.py` composes; it does not
  require altering `config.MODE_LABELS`'s existing middle-dot character
  (`"Mode 1 · No Retrieval"`), which is neither an emoji nor an em/en dash and is
  displayed as-is.
- This no-emoji/no-dash constraint applies to **user-visible strings only** — it does
  not apply to Python code comments (of which `app.py` should have few to none, per
  this repo's default no-comments convention).
- The three preset example questions must be these exact strings (already
  live-verified in Phase 5 to route to their respective modes):
  - `"What is a vector database?"` (Mode 1)
  - `"According to the BERT paper, what pretraining tasks does BERT use?"` (Mode 2)
  - `"Compare how BERT and the Transformer paper each handle positional information, and explain any tradeoffs."` (Mode 3)
- `graph.invoke(...)` is always called with `trace: []` explicitly set in the initial
  state dict, matching the exact path already verified live in Phase 5.
- `build_graph()` is called once per script run via a module-level `graph = build_graph()`
  (not per-button-click). Streamlit reruns the whole script on every interaction, so
  this still recompiles once per rerun rather than persisting across reruns — that is
  acceptable for a demo of this size per the Phase 5 final review's note that graph
  compilation is cheap; `st.cache_resource` would avoid the per-rerun recompilation
  but is an optimization out of scope for this phase, not a correctness requirement.

---

### Task 1: `app.py` — layout, input, and graph invocation

**Files:**
- Create: `app.py` (repo root)

**Interfaces:**
- Consumes: `src.graph.build_graph() -> CompiledStateGraph` (Phase 5, already built),
  whose `.invoke(state: dict) -> dict` returns a dict with keys matching
  `src.state.RAGState`: `question`, `route`, `mode_label`, `sub_questions`,
  `retrieved`, `answer`, `citations` (`list[dict]`, each dict has `document_name`,
  `page_number`, `chunk_id`, `section_heading`, `score`, `marker`), `critique_clean`,
  `unsupported_claims`, `iterations`, `trace` (`list[str]`).
- Produces: nothing consumed by later tasks — this is the final phase's only file.

There is no TDD cycle for this task: Streamlit apps have no meaningful unit-testable
logic in `app.py` itself (it's pure UI composition calling an already-tested graph),
so this task is written directly, then verified by running the app (Task 2).

- [ ] **Step 1: Write `app.py`**

```python
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
    with st.spinner("Running the adaptive RAG pipeline..."):
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
```

- [ ] **Step 2: Verify the file has no syntax errors**

Run: `cd /Users/apple/Desktop/Projects/aws-rag && source venv/bin/activate && python -c "import ast; ast.parse(open('app.py').read())"`
Expected: no output, exit code 0 (confirms valid Python syntax before launching
Streamlit).

- [ ] **Step 3: Scan for forbidden characters in user-visible strings**

Run:

```bash
cd /Users/apple/Desktop/Projects/aws-rag && python3 -c "
import re
text = open('app.py', encoding='utf-8').read()
em_en_dashes = re.findall(r'[–—]', text)
emoji = re.findall(r'[\U0001F300-\U0001FAFF☀-➿]', text)
print('em/en dashes found:', em_en_dashes)
print('emoji found:', emoji)
"
```

Expected: `em/en dashes found: []` and `emoji found: []`. If either list is
non-empty, find the offending string in `app.py` and replace it (hyphen `-` for
dashes; remove/reword for emoji) before proceeding.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "Add Streamlit UI calling the compiled graph directly"
```

---

### Task 2: Live browser verification of all three modes and the error path

**Files:**
- None created/modified — this task runs the app in a real browser, observes
  behavior directly, then records results in `BUILD_LOG.md` and flips Phase 6's
  checkbox in `PLAN.md`.

**Interfaces:**
- Consumes: `app.py` from Task 1; the real Groq API (`GROQ_API_KEY` in `.env`,
  already confirmed working); the real FAISS index (already built in Phase 2).

This task requires driving a real browser against a running Streamlit server, so
it is a manual verification task, not a TDD unit-test task — matching how every
prior phase's live verification was done, and per the guidance that UI changes must
be exercised in an actual browser before being called done, not just confirmed via
code reading or type-checking.

- [ ] **Step 1: Launch the app**

Run: `cd /Users/apple/Desktop/Projects/aws-rag && source venv/bin/activate && streamlit run app.py --server.headless true &`

Expected: Streamlit prints a local URL (typically `http://localhost:8501`). Open it
in a browser (or use a browser-automation tool if available) to visually confirm the
page loads: title "Adaptive RAG" visible, the example-question selectbox, the
free-text input pre-filled with the first preset's question, and a "Run" button.

- [ ] **Step 2: Exercise Mode 1 (the vector-database example)**

In the browser: select "Mode 1 example - general knowledge" from the dropdown
(confirms the text input updates to "What is a vector database?"), click Run,
wait for the spinner to resolve.

Expected, observed directly in the rendered page: a colored badge reading
"Mode 1 - No Retrieval" (or whatever `config.MODE_LABELS[ROUTE_EASY]` currently
renders as with the middle-dot), an answer paragraph about vector databases, no
"Citations" subheader (since Mode 1 has zero citations), and an expander that when
opened shows the trace with exactly two entries (`router -> ...`,
`direct_answer (no retrieval)`) and no iteration/critique lines (since `route` isn't
`"hard"`).

- [ ] **Step 3: Exercise Mode 2 (the BERT pretraining-tasks example)**

Select "Mode 2 example - single document lookup", click Run, wait for the spinner.

Expected, observed directly: a distinctly-colored badge reading
"Mode 2 - Single-Hop Retrieval", an answer mentioning masked LM and next sentence
prediction, a "Citations" subheader listing at least one citation into `bert.pdf`
with a page number, and the expander showing a trace with `router`, `search`,
`generate` entries (no `decompose`/`critique`) and no iteration/critique lines shown
(since `route` isn't `"hard"`).

- [ ] **Step 4: Exercise Mode 3 (the BERT-vs-Transformer comparison example)**

Select "Mode 3 example - multi-hop comparison", click Run, wait for the spinner
(this one may take longer - up to 3 search/generate/critique cycles).

Expected, observed directly: a distinctly-colored badge reading
"Mode 3 - Multi-Hop + Self-Critique", a multi-paragraph answer comparing both
papers, a "Citations" subheader listing citations from both `bert.pdf` and
`attention_is_all_you_need.pdf`, and the expander showing "Iterations: " followed by
a number between 1 and 3 inclusive, a "Critique clean:" line, and a trace with
multiple `search`/`generate`/`critique` entries (matching the pattern already
observed in Phase 5's live dry-run: up to 3 `search` entries, never more).

- [ ] **Step 5: Exercise free-text entry**

Clear the text input and type a new question not in the preset list (e.g. "What is
the capital of France?"), click Run, wait for the spinner.

Expected: the app runs the same as any preset question - a mode badge, an answer, and
(if applicable) citations and trace appear correctly for this fresh, non-preset input.

- [ ] **Step 6: Visually confirm no emoji or em/en dash appears anywhere on the rendered page**

Across all four runs above (Steps 2-5), visually inspect the rendered page for any
emoji, pictographic character, or em/en dash in the title, captions, labels, badge
text, answer text is not itself covered by this constraint (it's model-generated
content and out of scope), or trace/expander content. Confirm none appear in any
`app.py`-authored string (title, caption, selectbox label, badge label, "Citations"
subheader, expander title, "Iterations:"/"Critique clean:"/"Trace:" labels).

- [ ] **Step 7: Exercise the error path**

Temporarily break the Groq key to trigger a real failure: in a separate terminal,
run `cd /Users/apple/Desktop/Projects/aws-rag && cp .env .env.bak && sed -i '' 's/GROQ_API_KEY=.*/GROQ_API_KEY=invalid_key_for_error_test/' .env`.
Restart the Streamlit app (stop it and re-run the Step 1 command so it picks up the
changed `.env`), click Run on any preset question.

Expected: `st.error(...)` shows a red error box with a message derived from the
exception (e.g. a Groq authentication error) - not Streamlit's default full
traceback page.

Then restore the real key: `cd /Users/apple/Desktop/Projects/aws-rag && mv .env.bak .env`,
and restart the app once more to confirm it works normally again (re-run Step 2 or
3 briefly to confirm recovery).

- [ ] **Step 8: Stop the Streamlit server**

Run: `pkill -f "streamlit run app.py"` (or stop the background process started in
Step 1 by its job/process ID).

- [ ] **Step 9: Update `PLAN.md`**

Read `PLAN.md`, find `## Phase 6 — Streamlit UI (\`app.py\`)  ☐` and its body. Change
the checkbox to `☑`, and add the live browser verification results underneath,
following the same style as Phases 2-5's updates (status line + concrete observed
results per mode, the error-path confirmation, and the emoji/dash scan result).

- [ ] **Step 10: Add a `BUILD_LOG.md` entry**

Append a new entry (next sequential number after entry 10) documenting: what was
built (`app.py` - layout, preset+free-text input, spinner/error handling, mode
badge, citations, collapsed trace expander), why (Phase 6 per PLAN.md - the last
piece needed to demo the project), the live browser verification results (Task 2 -
exact observations per mode, the error-path test, the emoji/dash scan), and the
explicit no-emoji/no-em-dash constraint and why it was imposed (a deliberate
stylistic choice, per the user).

- [ ] **Step 11: Commit**

```bash
git add PLAN.md BUILD_LOG.md
git commit -m "Complete Phase 6: Streamlit UI verified live in browser"
```

---

## Self-Review Notes

- **Spec coverage:** layout (title, preset dropdown + free-text, Run button) - Task
  1. Spinner + try/except error handling - Task 1. Mode badge, answer, citations,
  collapsed expander with iteration/critique/trace - Task 1. No-emoji/no-em-dash
  constraint - enforced by an automated scan in Task 1 Step 3 (not just visual
  inspection) plus a manual visual check in Task 2 Step 6, so the requirement is
  checked twice, once mechanically and once by observation. Live browser
  verification of all 3 modes plus free-text plus error path - Task 2. All spec
  sections covered.
- **Placeholder scan:** no TBD/TODO; the one step with narrative structure (Task 2
  Step 6) still ends with a concrete, checkable list of exactly which strings to
  inspect.
- **Type consistency:** `build_graph()` and its `.invoke(dict) -> dict` return
  contract match exactly what Phase 5's `src/graph.py` already provides (no
  reinterpretation). Dict key names used in `app.py` (`route`, `mode_label`,
  `answer`, `citations`, `critique_clean`, `iterations`, `trace`, and the citation
  dict's `marker`/`document_name`/`page_number`) match `src/state.py`'s `RAGState`
  and `src/nodes/generate.py`'s citation-dict construction exactly.
