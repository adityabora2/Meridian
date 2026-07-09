# BUILD LOG — Adaptive RAG (local)

Append-only audit log. One entry per meaningful step: file created, decision made,
dependency installed, bug hit + resolved, test run. Factual and brief.

---

### 1 — Project scaffold
- **What:** Created directory tree (`data/papers/`, `src/`, `src/nodes/`, `tests/`),
  `requirements.txt`, `.env.example`, `.gitignore`, `src/config.py`, this log.
  Initialized a git repo.
- **Why (decisions):**
  - Built at repo root (`aws-rag/`) rather than a nested `adaptive-rag-local/` folder,
    per user's choice — one repo, simpler paths.
  - **One LLM model for all four LLM steps** (`llama-3.3-70b-versatile` on Groq).
    Routing/decompose/generate/critique differ by *prompt*, not model. Keeps the demo
    explainable and cheap.
  - **`temperature=0.0`** — routing and critique are classification tasks; we want
    stable, reproducible decisions for the interview demo, not creativity.
  - **FAISS persisted to `index/`** (index + a JSON metadata sidecar) so we don't
    re-embed on every run, per spec.
- **Files:** `requirements.txt`, `.env.example`, `.gitignore`, `src/config.py`,
  `src/__init__.py`, `src/nodes/__init__.py`, `BUILD_LOG.md`
- **Deviation from spec:**
  - Added `.gitignore` (not in spec's file list) to keep secrets/index/PDFs out of git.
    Minor, standard hygiene.

---

### 2 — Dependencies installed
- **What:** Created `venv/` (Python 3.13.7), installed all of `requirements.txt`.
  Smoke-tested imports: PyMuPDF 1.25.1, faiss-cpu, langgraph 0.2.60, groq 0.13.1,
  sentence-transformers 3.3.1, streamlit 1.41.1 — all import cleanly.
- **Why:** Pinned versions chosen to be mutually compatible on Python 3.13. torch 2.12
  pulled in as a sentence-transformers dependency (CPU wheel, ~88 MB) — expected.
- **Files:** none (environment only).
- **Deviation:** none.

---

### 3 — Chunk size decision (D1): 256 tokens instead of spec's 512
- **What:** Set `CHUNK_SIZE_TOKENS = 256` (was 512), kept 50-token overlap, and bumped
  `TOP_K` from 4 to 6.
- **Why:** `all-MiniLM-L6-v2` truncates input at 256 tokens. A 512-token chunk would be
  only half-embedded — the vector would silently ignore the back half of every chunk,
  degrading retrieval honesty. Choosing 256 makes the embedding represent the whole
  chunk. Downsides (≈2× more chunks, larger index) are trivial at this corpus size.
  `TOP_K` raised to 6 so the generate/multi-hop steps still see enough context now that
  each chunk carries less text. Decision made with the user (they chose 256 over 512).
- **Files:** `src/config.py`.
- **Deviation from spec:** Yes — spec said 512 tokens; we use 256. Deliberate, discussed,
  and documented here. This is the only functional deviation so far.

---

### 4 — Phase 0/1 re-scaffold after non-persisted writes
- **What:** An earlier scaffold pass reported success but the files did not persist to
  disk (only `PLAN.md`, `src/__init__.py`, `src/nodes/__init__.py`, and the `venv/`
  survived). Re-created `requirements.txt`, `.env.example`, `.gitignore`, `src/config.py`,
  `BUILD_LOG.md`, and `data/papers/`. venv + installed deps were intact and re-verified
  (`import faiss, langgraph` → ok).
- **Why:** Recover a clean, on-disk Phase 0 baseline before starting Phase 2.
- **Files:** re-created as listed above.
- **Deviation:** none (recovery of prior state).

---

### 5 — Phase 2: ingestion pipeline (`src/ingest.py`)
- **What:** Built the full offline pipeline: PyMuPDF parse → best-effort section-heading
  detection → 256/50 token chunking → local `all-MiniLM-L6-v2` embeddings → FAISS
  `IndexFlatIP` → persist to `index/faiss.index` + `index/metadata.json`. Added runtime
  helpers `load_index()` and `search(query, k)` (imported later by the search node — this
  is deviation D2: no separate `retriever.py`). Added a `--self-test` mode that builds a
  synthetic 3-section PDF and asserts parse→chunk→embed→persist→reload→search works.
- **Why (decisions):**
  - **Token counting via the embedder's own tokenizer**, so "256 tokens" == exactly the
    256-token window MiniLM sees. Chunk boundaries line up with the embedding window.
  - **Chunk by character offsets (`return_offsets_mapping`), not by decoding token IDs.**
    MiniLM's WordPiece tokenizer lowercases and mangles punctuation on decode
    ("self-critique" → "self - critique"); slicing the original string keeps chunk text
    verbatim — important for citations and for the LLM's context.
  - **Normalized embeddings + `IndexFlatIP`** so inner product == cosine similarity.
    Flat index is exact and simple; fine for a handful of papers.
  - **Heading detection via `get_text("blocks")` + regex** (numbered headings + common
    section keywords). Best-effort per spec; font-size analysis would be more precise but
    heavier. All 3 synthetic sections detected correctly.
- **Bugs hit + resolved (2):**
  1. *Lossy chunk text.* First implementation decoded token IDs back to text → stored
     chunks were lowercased/de-punctuated. Fixed by switching to offset-based slicing of
     the original string.
  2. *Truncated synthetic PDF text.* The self-test's synthetic PDF used `insert_text` at a
     fixed point; long paragraphs ran off the page and PyMuPDF only extracted the visible
     portion, so chunk tails were lost at parse time. Fixed by generating the synthetic
     PDF with `insert_textbox` (wraps within a rect). Note: this was a test-fixture flaw,
     not a chunking bug — real papers lay out text properly.
- **Test result:** `python -m src.ingest --self-test` → PASSED. 3 chunks, all headings
  detected, max chunk 60 tokens (≤256 window), reload+search returns the correct section
  as top hit (score 0.586). Empty-corpus run gives a clear "drop PDFs and re-run" error.
- **Files:** `src/ingest.py` (new).
- **Deviation from spec:** D2 (retrieval helpers live in `ingest.py`, no `retriever.py`) —
  keeps the spec's file list exact. D1 (256-token chunks) already logged in entry 3.

---

### 6 — Phase 3: router + Mode 1/2 nodes
- **What:** Built the routing node and the nodes that make Mode 1 and Mode 2 work
  end-to-end (spec's priority modes):
  - `src/nodes/llm.py` — one shared Groq client/`chat()` helper for all nodes; raises a
    clear `LLMConfigError` when the key is missing.
  - `src/state.py` — the `RAGState` TypedDict all nodes share (question, route, mode_label,
    sub_questions, retrieved, answer, citations, critique fields, iterations, trace).
  - `src/nodes/router.py` — Groq classify easy/medium/hard, defensive label parsing,
    fail-safe to "medium" (retrieve) when the reply is unparseable.
  - `src/nodes/direct_answer.py` — Mode 1: answer directly, zero retrieval, no citations.
  - `src/nodes/search.py` — FAISS top-k via `ingest.search`; single-hop (Mode 2) or one
    search per sub-question (Mode 3); pools + dedups by chunk_id; **bumps the iteration
    counter** (the value the Phase-5 conditional edge reads).
  - `src/nodes/generate.py` — numbered-evidence prompt, inline [n] citations, resolves
    cited markers back to structured citations (document + page) for the UI.
- **Why (decisions):**
  - **Shared `llm.py` helper** so every node is "same model, different prompt" — the
    story the demo tells. Also centralizes the missing-key error.
  - **Router fails safe toward retrieval** (medium) on parse failure: a wrong retrieve is
    cheaper than a confident, ungrounded no-retrieval answer.
  - **`RAGState` in its own module** so nodes type-check against one contract and field
    names never drift between node files and `graph.py`.
  - **Search accumulates + dedups** rather than replacing, so the Mode-3 loop adds new
    evidence across iterations instead of discarding prior hits.
  - **Citations resolved from the answer text** (`[n]` markers) so the UI shows exactly
    the sources the model used, not the whole retrieved set.
- **Tests (offline, no Groq key needed):** unit-tested every non-LLM path — router label
  parsing + fallback, search single/multi-hop pooling/dedup/iteration bump, generate
  citation extraction + no-evidence path. All PASSED. Also verified `llm.chat()` raises a
  clear `LLMConfigError` when the key is absent (the pre-`.env` state).
- **Not yet tested:** live Groq routing accuracy — needs `GROQ_API_KEY` in `.env`. Will
  validate in Phase 7 with the 30-question set once the key is present.
- **Files:** `src/nodes/llm.py`, `src/state.py`, `src/nodes/router.py`,
  `src/nodes/direct_answer.py`, `src/nodes/search.py`, `src/nodes/generate.py` (all new).
- **Deviation from spec:** Added `src/nodes/llm.py` and `src/state.py` (not in the spec's
  file list) — thin shared helpers, not features. They keep nodes DRY and the state
  contract single-sourced. No behavioral deviation.

---

### 7 — Phase 2 completion: real-PDF ingestion
- **What:** Populated `data/papers/` with two real papers from arXiv ("Attention Is All
  You Need", BERT) and ran `python -m src.ingest` (not `--self-test`) for the first time.
  Indexed 132 chunks (50 + 82) from the two PDFs into `index/faiss.index` +
  `index/metadata.json`. Verified retrieval with a real query ("What is self-attention and
  how is it computed?") — top-3 hits were on-topic with correct document_name/page.
- **Why:** Phase 2's exit check required proving the pipeline on real PDFs, not just the
  synthetic self-test fixture, before Phase 3's nodes could be considered fully validated
  against real data.
- **Bug/limitation found:** The heading-detection heuristic (regex + numbered-heading
  matching over `get_text("blocks")`) mislabels most sections as "Abstract" on real
  academic PDF layouts — it does not generalize from the synthetic 3-section test.
  Chunk text, embeddings, and retrieval ranking are unaffected; only the
  `section_heading` metadata field is unreliable. Left as a known issue — not blocking
  Mode 1/2, revisit if citation display needs accurate section names.
- **Files:** none (data only: `data/papers/attention_is_all_you_need.pdf`,
  `data/papers/bert.pdf`; regenerated `index/faiss.index`, `index/metadata.json`).
- **Deviation:** none.

---

### 8 — Phase 3 completion: live Groq routing verified
- **What:** Confirmed the embedding model (`all-MiniLM-L6-v2`) was already cached locally
  (`~/.cache/huggingface/hub/`, loads with no network/key needed). Created `.env` from
  `.env.example` and added a real `GROQ_API_KEY` (confirmed gitignored — `git check-ignore`
  passes). Ran `route_question()` live against Groq for 3 questions spanning all three
  route labels; all classified correctly (easy / medium / hard).
- **Why:** Phase 3's exit check for router accuracy was blocked on having a real Groq key;
  this closes that gap with a spot-check ahead of the full Phase 7 test set.
- **Files:** `.env` (new, gitignored, not committed).
- **Deviation:** none.

---

### 9 — Phase 4: decompose + critique nodes (Mode 3 building blocks)
- **What:** Built the two remaining nodes for Mode 3 (multi-hop + self-critique), per
  the approved design spec (`docs/superpowers/specs/2026-07-10-phase4-mode3-nodes-design.md`)
  and implementation plan (`docs/superpowers/plans/2026-07-10-phase4-mode3-nodes.md`),
  executed via subagent-driven development (fresh implementer + reviewer per task):
  - `src/nodes/decompose.py` — splits a hard question into 2-3 sub-questions, one per
    line, capped at 3 even if the LLM over-produces, falls back to `[question]` if
    parsing yields nothing.
  - `src/nodes/critique.py` — checks the draft answer's claims against the numbered
    evidence (reuses the same `_format_evidence` numbering `generate.py` uses, duplicated
    locally rather than shared — a deliberate spec decision, not a DRY miss). Uses
    sentinel-line output (`VERDICT: clean` / `VERDICT: unsupported` + `CLAIMS:` bullets)
    matching `router.py`'s plain-text house style, not JSON. Falls back to `clean` on any
    unparseable response — a parse bug shouldn't force the Phase-5 loop to spin.
    Unsupported claims are phrased as short search-friendly questions (not verbatim
    quotes) so Phase 5 can feed them back into `search_node` as the next iteration's
    queries.
- **Why:** Phase 4 per PLAN.md; unblocks Phase 5's graph assembly, which needs both
  nodes' output fields (`sub_questions`, `critique_clean`, `unsupported_claims`) to wire
  the conditional edge.
- **Tests:** Offline parsing unit tests for both nodes (no Groq needed) — decompose's
  newline-splitting/cap/fallback (5 cases) and critique's sentinel-line parsing across 5
  branches (clean, unsupported+claims, unparseable→fallback, empty→fallback,
  unsupported-with-no-claims→fallback) — all **PASSED**. Both task diffs were reviewed
  independently (spec compliance + code quality); both **Approved**, no Critical/Important
  findings.
- **Live spot-checks:** `decompose()` on "Compare how BERT and the Transformer paper each
  handle positional information, and explain any tradeoffs." returned 3 clean,
  independently-searchable sub-questions. `critique()` on an answer with one grounded
  claim (Adam optimizer) and one fabricated claim ("trained on 8 V100 GPUs for 12 days")
  correctly returned `unsupported` with the fabricated claim flagged (and only that one).
  `critique()` on a fully-grounded answer returned `clean` with no flagged claims.
- **Files:** `src/nodes/decompose.py`, `src/nodes/critique.py`, `tests/test_decompose.py`,
  `tests/test_critique.py` (all new).
- **Deviation:** none.

---

### 10 — Phase 5: graph assembly (`src/graph.py`) — the core deliverable
- **What:** Wired all six existing nodes (`router`, `direct_answer`, `search`,
  `generate`, `decompose`, `critique` — none modified) into a compiled LangGraph
  `StateGraph`, executed via subagent-driven development (branch functions, then full
  assembly, then live verification, each independently reviewed):
  - `route_from_router(state) -> str` — reads `state["route"]`, returns
    `"direct_answer"` / `"search"` / `"decompose"`.
  - `route_from_critique(state) -> str` — reads `state["critique_clean"]` and
    `state["iterations"]` against `config.MAX_ITERATIONS`; returns `"prepare_retry"`
    or `"end"`. Boundary-tested: `iterations == 2` (→ retry allowed) vs.
    `iterations == 3` (→ cap hit, ends) — the single most safety-critical branch in
    the whole phase, since it's what guarantees the Mode 3 loop terminates.
  - `prepare_retry(state) -> RAGState` — a new one-line glue node, inline in
    `graph.py` (not under `src/nodes/`, since it makes no Groq call): sets
    `sub_questions = unsupported_claims` so `search_node` searches specifically for
    the evidence critique found missing, rather than blindly repeating the original
    sub-questions.
  - `build_graph() -> CompiledStateGraph` — registers all 7 nodes (6 existing + 1
    new), wires router's conditional edge, the linear Mode 2 path, the Mode 3 path,
    an inline `generate`→`critique`-or-`END` branch keyed on whether
    `sub_questions` is set (distinguishes Mode 2 from Mode 3 without adding new
    state), and critique's conditional retry/end edge.
- **Why:** Phase 5 per PLAN.md — described as "the core deliverable." Unblocks Phase
  6 (Streamlit UI, which calls the compiled graph directly).
- **Design decisions:**
  - **Plain `TypedDict` overwrite semantics, no `Annotated` reducers** on any
    `RAGState` field. Every node already does its own read-modify-write internally
    (e.g. `search_node` returns the full pooled+deduped list, not just new hits) —
    adding a reducer would double-accumulate on top of that.
  - **No new iteration counter.** `search_node` (Phase 3, unchanged) already
    increments `state["iterations"]` on every call, and the router (unchanged)
    already resets it to `0` per run — the cap check just reads the existing field.
  - **`prepare_retry` kept out of `src/nodes/`** since it's pure graph-wiring glue,
    not a node with independent LLM-calling responsibility — keeps the `src/nodes/`
    directory reserved for actual Groq-calling nodes.
- **Tests:** Offline branch-function unit tests (7 cases: all 3 router branches, plus
  clean/retry/cap-hit/boundary cases for critique, including the safety-critical
  `iterations == MAX_ITERATIONS - 1` vs. `== MAX_ITERATIONS` boundary) — all
  **PASSED**. Both task diffs independently reviewed (spec compliance + code
  quality); both **Approved**. Task 2's reviewer specifically traced the full
  `prepare_retry → search → generate → critique` cycle against `search.py` and
  `critique.py` to confirm no dangling edges, correct use of the `END` sentinel
  object (not a string), and correct iteration-counter accumulation across cycles —
  no Critical/Important findings.
- **Live end-to-end verification (all 3 modes):**
  - **Mode 1** — "What is a vector database?" → routed `easy`, answered directly, zero
    citations, trace = `router → direct_answer` only.
  - **Mode 2** — "According to the BERT paper, what pretraining tasks does BERT use?"
    → routed `medium`, correctly answered MLM + NSP with a citation into `bert.pdf`
    page 8, `iterations: 1`, trace = `router → search → generate` only (no
    decompose/critique).
  - **Mode 3** — "Compare how BERT and the Transformer paper each handle positional
    information, and explain any tradeoffs." → routed `hard`, looped
    `decompose → search → generate → critique` then
    `prepare_retry → search → generate → critique` **twice more**, hit the cap at
    exactly `iterations: 3` (trace shows exactly 3 `search` entries, never more),
    terminated with `critique_clean: False` (expected cap-hit termination — the
    graph returned a result instead of hanging or looping past the cap) and a
    6-citation answer spanning both `bert.pdf` and `attention_is_all_you_need.pdf`.
- **Bug found + fixed during final whole-branch review:** none for this phase (Phase
  4's asymmetric-bullet-stripping fix was the prior phase's finding). This phase's
  final review found only cosmetic Minor notes (path_map key-naming asymmetry,
  file now holding several small pieces) — no fixes required.
- **Files:** `src/graph.py`, `tests/test_graph.py` (both new).
- **Deviation:** none.

---

### 11 — Phase 6: Streamlit UI (`app.py`)
- **What:** Built a single-page Streamlit app calling `build_graph()` (Phase 5,
  unchanged) directly — no API layer. Layout: title, a preset-question dropdown (one
  example per mode, reusing the exact questions already live-verified in Phase 5)
  plus a free-text input defaulting to the preset's value, a Run button. On success:
  a colored mode badge (green/blue/gold for easy/medium/hard) as the visual headline,
  the answer, a Citations list (document + page), and a collapsed
  "How this answer was produced" expander with iteration count, critique verdict
  (Mode 3 only), and the full trace. On failure: `st.error(...)` with the exception
  message, not a raw traceback.
- **Why:** Phase 6 per PLAN.md — the last piece needed to actually demo the project
  end-to-end in a browser.
- **Explicit user constraint:** no emoji or pictographic characters, and no em/en
  dashes, in any user-visible string in `app.py` (regular hyphens only) — a
  deliberate stylistic choice to avoid text that reads as AI-generated boilerplate.
  Does not apply to Python comments or to pre-existing strings from other files
  (e.g. `config.MODE_LABELS`'s middle-dot, or the `→`/`×` characters in node trace
  strings from Phases 3-4 — neither is an emoji nor an em/en dash).
- **Tests:** A syntax check plus an automated emoji/dash scan were run as part of
  implementation. The task reviewer independently re-verified the emoji/dash
  constraint character-by-character rather than trusting the self-reported scan, and
  found the file contains **zero non-ASCII characters at all** — a stronger and more
  conclusive result than a regex match. No Critical/Important findings.
- **Live browser verification:** No browser-automation tool was available in-session
  (no Playwright/Selenium, no local Chrome/Chromium). Installed Playwright plus
  headless Chromium as a one-time verification tool (not added to `requirements.txt`
  — it's a dev-time verification aid, not a runtime dependency of the app) and drove
  the running app end-to-end:
  - **Mode 1** ("What is a vector database?"): green "Mode 1 · No Retrieval" badge,
    correct answer, no Citations section (correct — Mode 1 has none), 2-line trace
    in the expander.
  - **Mode 2** ("According to the BERT paper, what pretraining tasks does BERT
    use?"): blue "Mode 2 · Single-Hop Retrieval" badge, correct MLM/NSP answer, one
    citation into `bert.pdf` page 8, trace = `router → search → generate` only.
  - **Mode 3** ("Compare how BERT and the Transformer paper..."): gold "Mode 3 ·
    Multi-Hop + Self-Critique" badge, a multi-paragraph comparison answer, 5
    citations spanning both `bert.pdf` and `attention_is_all_you_need.pdf`,
    `Iterations: 3` (hit the cap — consistent with Phase 5's prior live result on the
    same question), `Critique clean: False` (expected cap-hit outcome), and an
    11-step trace rendered correctly in the expander.
  - **Free-text entry** ("What is the capital of France?", not a preset): routed to
    Mode 1, answered correctly about Paris — confirms free-text works identically to
    the preset dropdown path.
  - **Error path:** temporarily set `GROQ_API_KEY` to an invalid value, restarted the
    app, clicked Run, and confirmed `st.error(...)` rendered a clean red error box
    ("Something went wrong: Error code: 401 - Invalid API Key") rather than
    Streamlit's default traceback page. Restored the real key, restarted, and
    confirmed the app answers correctly again.
  - Screenshots taken at each step confirmed all of the above visually, not just via
    extracted page text.
- **Files:** `app.py` (new, repo root).
- **Deviation:** none in the shipped code. Process deviation: Playwright + headless
  Chromium were installed mid-task to satisfy the "verify in a real browser"
  requirement, since no such tool was pre-installed in this environment — a
  one-time addition to the local dev environment, not to the project's own
  dependencies.
