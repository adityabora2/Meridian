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

---

### 12 — Corpus expansion to 10 papers, production-readiness audit, and Groq retry hardening
- **What:** Three related pieces of work, done together:
  1. **Corpus expanded from 2 to 10 papers.** Added GPT-3, RoBERTa, T5, ALBERT,
     ELECTRA, DistilBERT, XLNet, PaLM (arXiv PDFs) alongside the existing Attention
     Is All You Need and BERT. Re-ran `python -m src.ingest`: 1490 chunks indexed (up
     from 132). Spot-checked retrieval across 3 of the new papers (T5, ELECTRA,
     DistilBERT) — all returned correct, on-topic top hits.
  2. **Full production-readiness audit**, prompted by the user calling the project
     "so-called production ready." Read every source file and independently
     verified BUILD_LOG's own prior claims rather than trusting them. Key findings:
     - FAISS `IndexFlatIP` measured at 0.036ms mean search time at 1490 chunks —
       not a real concern until roughly 1M+ chunks; explicitly not "fixed."
     - Heading-detection heuristic (flagged as a known issue in entry 7) is worse
       than previously described: **86% of all 1490 chunks** are mislabeled
       "Abstract"/"References"/"Acknowledgments" — measured directly against
       `index/metadata.json`, not estimated.
     - Cross-paper retrieval contamination is a new, real risk at 10 papers:
       `search.py` pools FAISS hits with zero document-level filtering, and the 10
       papers are vocabulary-similar (BERT/RoBERTa/ALBERT/ELECTRA/DistilBERT), so a
       question about one paper can now surface chunks from a sibling paper ranked
       higher — feeding potentially off-topic evidence into Mode 3's critique loop.
     - `router.py`/`decompose.py`'s few-shot prompt examples still reference
       "Adaptive-RAG, Self-RAG, Chain-of-Verification" — papers not in the corpus at
       all. Low-impact (routing is about question shape, not topic) but a real
       prompt/corpus mismatch worth fixing.
     - `llm.py`'s `chat()` had no retry/backoff/timeout configuration of its own
       (though the Groq SDK does retry transient errors internally by default,
       `max_retries=2`). Mode 3 can make up to 8 Groq calls per question, making it
       the most rate-limit-exposed path — directly explaining the rate limit the
       user hit earlier this session.
     - BUILD_LOG entry 6 claims router/search/generate were "unit-tested, all
       passed" — verified **no test files exist** for any of the three. Only
       critique, decompose, and graph branch-function tests are actually committed
       and repeatable. This entry corrects that record.
  3. **Retry/timeout hardening**, the first of the audit's 5 prioritized fixes.
     Widened the Groq client's retry budget (`max_retries=5`, up from the SDK
     default of 2) and added an explicit `timeout=30.0`s in `src/nodes/llm.py`'s
     `_client()`. Deliberately did **not** add custom retry/backoff logic on top —
     the SDK's own mechanism, now with more room, is sufficient, and after it's
     exhausted the failure should propagate loudly (per the user's explicit
     decision) rather than be masked with a degraded/fake answer.
- **Why:** The user's own real Groq rate limit, hit mid-session, was the direct
  trigger. The audit was requested to get an honest read on "production ready"
  before continuing to Phase 7 against the larger, more realistic corpus.
- **Live verification of the fix:** Confirmed the constructed Groq client actually
  carries the new `max_retries=5`/`timeout=30.0` values. Re-ran the Phase 6
  error-path test (invalid API key) — still fails fast and cleanly (a 401 doesn't
  retry, correctly). Then, while testing the happy path, **hit a genuine live daily
  quota exhaustion** (`RateLimitError` 429: "tokens per day (TPD): Limit 100000,
  Used 99846... try again in 13m51s") — this is exactly the "sustained failure,
  fail loudly" case the design targeted, and it worked exactly as intended: a clear
  exception with the exact reset time embedded, ready to surface via `app.py`'s
  existing `st.error(...)` path. This confirms the fix is correct: retries cannot
  and should not paper over an exhausted daily budget, only transient bursts.
- **Not yet done (remaining audit items, explicitly deferred):** cross-paper
  retrieval contamination / document-scoped search, heading-detection quality,
  router/decompose prompt-corpus mismatch, a full live end-to-end pass of all 3
  modes against the new 10-paper corpus, and Phase 7's 30-question test set (paused
  pending the daily Groq quota reset).
- **Files:** `src/nodes/llm.py`, `app.py` (both modified). `data/papers/*.pdf` (8
  new, gitignored, not committed). `index/faiss.index`, `index/metadata.json`
  (regenerated, gitignored, not committed).
- **Deviation:** none from the approved retry-hardening spec. The corpus expansion
  and audit were user-directed additions to this session's scope, not part of any
  prior phase's plan.

---

### 13 — General document support (font-size heading detection, title extraction,
cross-document retrieval scoping, domain-agnostic prompts, papers→documents rename)
- **What:** A 6-task plan (`docs/superpowers/plans/2026-07-11-general-document-support.md`,
  design spec `docs/superpowers/specs/2026-07-11-general-document-support-design.md`)
  removing every academic-paper-specific assumption from the ingestion pipeline and
  LLM prompts, prompted by the user's push-back that a system meant for general PDF
  documents shouldn't be hardcoded to research-paper conventions:
  - **Task 1 — font-size-based heading detection** (`src/ingest.py`). Replaced the
    keyword-regex heuristic (only recognized academic section names, already
    measured at 86% wrong) with detection based on visual prominence: a heading is
    text whose font size is meaningfully larger than the document's detected
    body-text size. Caught and fixed a real calibration bug during implementation —
    the initial `_HEADING_SIZE_RATIO = 1.15` detected **zero** real section headings
    on the actual BERT paper (real academic headings are only ~10% larger than body
    text, not 15%+); recalibrated to `1.08` after sweeping against all 10 real
    papers in the corpus, confirmed via reverting the fix and observing the new
    regression tests genuinely fail.
  - **Task 2 — per-document title extraction** (`src/ingest.py`). New
    `document_title` field per chunk: PDF embedded metadata first, falling back to
    the largest-font text on page 1, falling back to the filename stem. Caught two
    more real-world bugs during required real-PDF verification: arXiv's rotated
    sidebar watermark being picked as the title (fixed by filtering to horizontal
    text only), and small-caps titles splitting one word across two font sizes
    (fixed by grouping spans into lines with a size-tolerance threshold instead of
    picking a single largest span). Both fixes independently verified by reverting
    them and confirming new regression tests fail without them.
  - **Task 3 — cross-document retrieval scoping** (`src/ingest.py`,
    `src/nodes/search.py`). New `match_document(text) -> str | None`: token-overlap
    scoring of a query against each document's title+filename, returning a
    confident match or `None` (safe default = search everything). `search()` gained
    an optional `document_hint` parameter to filter FAISS results to one document.
    Found the plan's originally-specified unweighted scoring was mathematically
    unsatisfiable (two of its own test fixtures required opposite outcomes from
    identical scores); fixed with a stopword filter plus filename-stem match
    weighting, keeping the plan's core scoring function untouched. Live-verified:
    an unfiltered search for a BERT question previously returned **zero** bert.pdf
    chunks in the top 5 (contaminated by roberta/electra/xlnet); with the fix
    applied, 100% bert.pdf.
  - **Task 4 — domain-agnostic prompts** (`src/nodes/router.py`,
    `src/nodes/decompose.py`, `src/nodes/generate.py`). Removed all references to
    "AI research papers (Adaptive-RAG, Self-RAG, Chain-of-Verification...)",
    replaced with domain-neutral framing and shape-based example questions. Fixed
    one stray "indexed papers" string in `generate.py`'s no-evidence fallback.
    Routing behavior confirmed unchanged (same easy/medium/hard classification on
    the standard 3 spot-check questions).
  - **Task 5 — renamed `papers` to `documents` throughout** (`src/config.py`,
    `src/ingest.py`, `.gitignore`, `data/papers/` → `data/documents/`).
    `PAPERS_DIR` → `DOCS_DIR`. Confirmed zero stray references remain anywhere in
    source. Also updated `.gitignore`'s PDF-ignore pattern to match the new path
    (needed — otherwise the old pattern would stop matching and a future broad
    `git add` could accidentally commit the corpus PDFs).
- **Why:** the user explicitly challenged the project's scope: "why is this project
  only handling and closing down on research papers... can't make this specific to
  research papers." PyMuPDF's actual capabilities (font/layout extraction) were
  never paper-specific — several hand-tuned heuristics and prompts were.
- **Tests:** every task went through implementer → task-reviewer cycles (subagent-
  driven development). Two tasks (1, 2) required a fix-and-re-review cycle after
  the controller independently caught real bugs the implementer's own synthetic
  tests missed — both times by checking behavior against the real 10-paper corpus
  rather than trusting narrow test fixtures. All offline tests
  (`tests/test_ingest_headings.py`, `tests/test_ingest_title.py`,
  `tests/test_ingest_match_document.py`, `tests/test_search.py`, plus every
  pre-existing test file) pass. `pytest` was installed as a dev-only tool (not
  added to `requirements.txt`) for the monkeypatch-based match_document/search
  tests, matching how Playwright was previously installed for browser verification.
- **Live verification (Task 6), completed in two parts due to a Groq quota block
  mid-task — see entry 14 for how it was ultimately completed:**
  - **Heading-detection accuracy, measured directly against the real 10-paper
    index:** generic/empty headings ("Abstract"/"References"/"Acknowledgments"/"")
    dropped from **86%** (the original audit finding, entry 12) to **18.5%**
    (289 of 1563 chunks) — a 4.6x reduction. The remaining top headings by
    frequency are genuine section names: "References", numbered "1 Introduction",
    "3 Results", specific appendix titles, etc. — not generic fallbacks.
  - **Document-matching spot-check:** 3 real queries against 3 different real
    papers (BERT, T5, ELECTRA) all correctly identified their target document.
  - **Mode 1/2 live end-to-end:** both correct — Mode 2's citations landed 100% on
    the correct document (`bert.pdf`), direct proof the contamination fix works on
    the live system, not just in isolated tests.
  - **Mode 3's full live run was blocked mid-task** by the Groq daily token quota
    (99,605/100,000 used, reset ~72 min out) — completed afterward using the new
    local LLM backend (Ollama/Qwen), see entry 14.
- **Files:** `src/ingest.py`, `src/nodes/search.py`, `src/nodes/router.py`,
  `src/nodes/decompose.py`, `src/nodes/generate.py`, `src/config.py`, `.gitignore`
  (all modified); `tests/test_ingest_headings.py`, `tests/test_ingest_title.py`,
  `tests/test_ingest_match_document.py`, `tests/test_search.py` (all new);
  `data/documents/` (renamed from `data/papers/`, gitignored PDFs unchanged in
  content, just relocated).
- **Deviation:** three implementation-level deviations from the plan's literal
  code, all disclosed, justified with real evidence, and independently verified by
  task reviewers (font-size ratio recalibration, title-extraction line-grouping
  instead of single-span, match_document's stopword+stem-weight scoring) — see
  each task's description above. No deviation from the plan's design intent.

---

### 14 — Replace Groq with local Qwen 2.5 7B via Ollama
- **What:** Per the user's explicit request ("I can't deal with the limit reset"),
  removed the Groq dependency entirely from `src/nodes/llm.py`, replacing it with a
  locally-served `qwen2.5:7b` model via Ollama (already installed on this machine).
  Since every LLM-calling node (`router.py`, `direct_answer.py`, `generate.py`,
  `decompose.py`, `critique.py`) already went through one shared `chat()` function,
  this was scoped to a single file plus config/dependency bookkeeping — no node
  file needed any change.
  - `src/nodes/llm.py`: `_client()` now returns the `ollama` module (after a
    reachability check via `ollama.list()`), rather than constructing a
    Groq-API-keyed client. `chat()`'s exact signature is unchanged; internally
    calls `ollama.chat(model=..., messages=..., options={"temperature":...,
    "num_predict":...})`. Removed the Groq-specific `max_retries=5, timeout=30.0`
    scaffolding — a local model has no rate limit to retry through, so a failure
    means Ollama isn't running or the model isn't pulled, and retrying wouldn't
    help either case. `LLMConfigError` now raises a clear message pointing at
    `ollama serve` / `ollama pull qwen2.5:7b`.
  - `src/config.py`: removed `GROQ_API_KEY`, `GROQ_MODEL`; added
    `OLLAMA_MODEL = "qwen2.5:7b"`. No API key needed for local inference.
  - `requirements.txt`: removed `groq`, added `ollama`.
  - `.env.example`: removed the Groq-key instructions; replaced with a note about
    `ollama serve`/`ollama pull qwen2.5:7b` one-time setup.
- **Why:** Groq's free-tier daily token quota (100,000 tokens/day) was repeatedly
  exhausted during this project's development and live verification — most
  recently blocking Task 6 of the general-document-support plan (entry 13) for
  ~72 minutes. The user wants this dependency and its recurring wait removed
  entirely, not managed around with a fallback/toggle.
- **Model choice:** `qwen2.5:7b` (~4.7GB, Q4 quantized) — comfortably fits this
  machine's 18GB RAM alongside the embedding model and FAISS index, and this
  project's four call types (single-word classification, short sub-question
  generation, cited-answer generation, claim verification) don't need a larger
  model's extra reasoning depth, especially at `temperature=0.0`.
- **Live verification:**
  - All pre-existing offline tests (`test_decompose.py`, `test_critique.py`,
    `test_graph.py`, `test_ingest_headings.py`, `test_ingest_title.py`,
    `test_ingest_match_document.py`, `test_search.py`) pass unchanged — none of
    them call `chat()` directly, confirming the swap didn't disturb any parsing/
    branching logic.
  - Router: the same 3 standard spot-check questions classify identically to
    every prior Groq-based verification in this project (easy/medium/hard).
  - Full graph, all 3 modes, run against the live 10-paper corpus (this also
    completes entry 13's Task 6, which was blocked on Groq's quota):
    - **Mode 1:** correct, no retrieval.
    - **Mode 2:** correct answer (BERT's MLM + NSP pretraining tasks), citations
      100% from `bert.pdf` — reconfirms the cross-paper contamination fix (entry
      13, Task 3) holds with the new LLM backend, not just with Groq.
    - **Mode 3:** graph terminates correctly (`iterations: 2`, within the cap,
      `critique_clean: True`), full decompose→search→generate→critique cycle
      confirmed working mechanically.
  - Error path: confirmed `chat()` raises a clear, catchable error when pointed at
    a nonexistent model name (`ResponseError: model 'nonexistent-model-xyz' not
    found`); the reachability-check path in `_client()` (Ollama daemon down) is
    implemented and code-reviewed but not forcibly tested live, since doing so
    would require stopping the user's persistent Ollama background service.
- **Known limitation found and accepted, not fixed:** Qwen 7B is less reliable
  than Groq's 70B model at following the `[n]` inline citation-marker format on
  Mode 3's longer, more complex generations. Reproduced twice: the model produced
  a well-formed, on-topic answer but used `(n1)`/`(n2)` style markers instead of
  `[1]`/`[2]`, so zero citations resolved even though the underlying evidence
  grounding was fine (confirmed separately: `generate()` in isolation with a
  single simple piece of evidence correctly produces `[1]`-style output every
  time — this is a complexity/reliability gap in the smaller model, not a bug in
  `generate.py`'s prompt or parsing). Decision (with the user): accept as a known,
  honest tradeoff of moving to a local 7B model rather than further prompt-
  engineering around it; a larger local model (14B) is a future option if this
  becomes a real problem in practice.
- **Files:** `src/nodes/llm.py`, `src/config.py`, `requirements.txt`,
  `.env.example` (all modified). No node file changed.
- **Deviation:** none from the approved spec.

---

### 15 — Phase 7 + production hardening (test set, README, corpus fix, coverage gaps, generalization proof)
- **What:** Cleared the full remaining action-item list to bring the system to a
  genuinely production-ready, accurate state. Six pieces:
  1. **Corpus data-accuracy fix.** The file named `gpt2.pdf` actually contained the
     GPT-3 paper ("Language Models are Few-Shot Learners" — an arXiv-ID mixup from
     the original download). Renamed it to `gpt3.pdf` and added the real GPT-2 paper
     ("Language Models are Unsupervised Multitask Learners") as `gpt2.pdf`. Corpus is
     now 11 correctly-labeled documents (1695 chunks). Also fixed `match_document`'s
     tokenizer to de-hyphenate model names ("GPT-2" → adds a "gpt2" token) so
     hyphenated names match their filename stem — previously gpt2/gpt3 tied on their
     near-identical titles and fell back to `None` (a recall miss). Regression tests
     added; BERT/T5 matches unaffected.
  2. **Closed the router/generate unit-test gap.** Added `tests/test_router.py`
     (label parsing, medium-fallback, iteration reset) and `tests/test_generate.py`
     (evidence formatting, citation extraction, out-of-range-marker rejection,
     no-evidence path) — both fully offline via monkeypatched `chat()`.
     **Correction to entry 6:** that entry claimed router/search/generate were
     "unit-tested, All PASSED," but until now no committed test file existed for
     router or generate (only search, via `test_search.py`). Those Phase-3 "tests"
     were one-off throwaway checks, never committed as regression tests. This entry
     sets the record straight and closes the gap for real. Full offline suite is now
     9 test files, all passing.
  3. **Phase 7 — routing-accuracy harness** (`tests/test_questions.py`). 30 labelled
     questions (10 easy / 10 medium / 10 hard) tuned to the actual 11-document
     corpus, run through the live router with per-class + overall accuracy, plus a
     `--mode3` flag measuring citation resolution on hard questions. **Live result on
     qwen2.5:7b: 28/30 = 93.3%** (easy 10/10, hard 10/10, medium 8/10 — the two
     medium misses classified as hard, which fails safe toward more retrieval).
  4. **Generalization proof.** Ingested a genuinely non-academic PDF (the US
     Constitution, 19 pages, 117 chunks) end-to-end. Font-size heading detection
     correctly identified legal structure it was never designed for ("Article V",
     "Amendment XII", "SECTION 3"); retrieval was accurate ("First Amendment" →
     establishment-of-religion text; "senators per state" → "two Senators from each
     State"). This closes the one open item from the general-document design spec's
     testing section — proof the work isn't secretly still paper-specific. The test
     surfaced (and fixed) a real robustness bug: the Constitution's embedded metadata
     title was junk ("constitution_pdf2"), and the metadata-first strategy trusted
     it; added `_looks_like_filename()` so junk/filename-shaped metadata is rejected
     in favor of font-based extraction. All 11 papers still extract correct titles.
  5. **README rewrite.** The prior README was stale (Groq, 512-token chunks,
     "research papers", "currently building"). Rewritten as an accurate, complete
     guide: Ollama-based setup with no API key, the general-document capability,
     correct stack, offline-vs-live test split, project structure, the measured
     93.3% routing accuracy, and an honest known-limitations section.
  6. **Working-tree cleanup.** Committed the four implementation plans under
     `docs/superpowers/plans/` (their specs were tracked, but the plans themselves
     never had been), the user's `INTERVIEW_PREP.md` notes, and a trailing-newline
     tidy on `direct_answer.py`.
- **Why:** the user asked to complete every outstanding action item and drive the
  system to "a ready and robust and accurate production-ready" state, using best
  judgment without per-item check-ins.
- **Known limitations, honestly stated (unchanged from entry 14, now documented in
  the README too):** Qwen-7B's inconsistent `[n]` citation formatting on long Mode-3
  generations (evidence retrieved correctly, markers sometimes unparseable) —
  tracked as a metric via `test_questions.py --mode3`, fixable with a larger local
  model. Cross-document scoping stays conservative (ambiguous queries search the
  whole corpus rather than risk wrong-document scoping).
- **Files:** `src/ingest.py` (de-hyphenation, junk-metadata title rejection);
  `tests/test_router.py`, `tests/test_generate.py`, `tests/test_questions.py` (new);
  `tests/test_ingest_match_document.py`, `tests/test_ingest_title.py` (new regression
  tests); `README.md` (rewritten); `PLAN.md`, `BUILD_LOG.md` (this entry);
  `docs/superpowers/plans/*` + `INTERVIEW_PREP.md` (committed); `data/documents/`
  (gpt2/gpt3 corrected, gitignored). Index regenerated (gitignored).
- **Deviation:** none — all work was within the stated goal of completing the
  outstanding action items to production-ready quality.

---

### 16 — Chatbot UI + three query-handling fixes
- **What:** First reworked the Streamlit UI from a preset-dropdown + Run-button layout
  into a proper chatbot (`st.chat_input`/`st.chat_message`, running history). The old
  dropdown was a preset-question picker, but sitting where the mode badge appears it
  repeatedly read as a mode *selector* and confused users into thinking they choose the
  retrieval mode — they don't, the router does. The mode is now a small caption on each
  answer ("Answered via single-hop retrieval") reporting the system's own decision. The
  flow is now: ingest documents → open UI → type a query → the system routes, retrieves,
  and answers on its own.
  Then live chatbot testing surfaced three real query-handling gaps, each fixed via a
  design-spec → plan → subagent-driven-development cycle (spec
  `docs/superpowers/specs/2026-07-11-query-handling-fixes-design.md`, plan
  `docs/superpowers/plans/2026-07-11-query-handling-fixes.md`):
  - **Fix A — corpus meta-questions.** "what documents are ingested?" had no capability
    and got routed to content retrieval (which correctly found nothing). Added a fourth
    router category `meta` (Task 1) and a new `corpus_info` node (Task 2, wired in Task
    3) that lists the indexed documents (name + title) directly from the index metadata,
    no LLM call. "what documents are ingested?" now routes `meta` and lists all 11.
  - **Fix B — first-page metadata retrieval.** "in xlnet who are the authors" returned
    nothing useful even though the author names ARE indexed — a raw author list on page
    1 ranks low semantically for "who are the authors". Added `page_one_chunks()` and,
    in `search_node`, when a query is confidently scoped to a document (via
    `match_document`), the document's page-1 chunks (title/authors/abstract) are boosted
    into the candidate set (Task 5). Bounded: only fires on a confident document match,
    so unscoped queries get no page-1 noise. The XLNet query now answers with the actual
    authors (Zhilin Yang, Zihang Dai, ...), cited to xlnet.pdf page 1.
  - **Fix C — document-aware routing.** "explain bert" answered from the model's own
    knowledge (route `easy`, no retrieval) instead of the BERT document. Now, when the
    router labels a question `easy` AND `match_document` finds it names an indexed
    document, the route upgrades to `medium` (Task 4). Strictly gated on `easy` — a
    `meta`/`medium`/`hard` label is never upgraded. "explain bert" now upgrades to
    medium and answers from bert.pdf with citations; genuine general-knowledge
    questions ("what is machine learning", which name no document) stay `easy`.
- **Why:** live use of the new chatbot showed the queries "weren't working" — the four
  reported cases were a UX confusion (mode dropdown) plus three distinct
  retrieval/routing gaps, all now fixed.
- **Tests:** every task went through implementer → task-reviewer cycles; all approved,
  no Critical/Important findings. New/updated tests: `test_corpus_info.py` (new),
  `test_router.py` (meta parsing + easy→medium upgrade + meta-not-upgraded),
  `test_graph.py` (meta→corpus_info), `test_search.py` (page-1 boost gated on scope).
  Full offline suite: **66 passing**.
- **Live verification (Task 6):** all four originally-failing queries confirmed fixed
  end-to-end through the full graph (documents-list via meta; XLNet authors named +
  cited to page 1; "explain bert" grounded in bert.pdf; "what is machine learning"
  correctly stays direct-answer). Routing-accuracy regression check held at **28/30 =
  93.3%** (unchanged — easy 10/10, medium 8/10, hard 10/10), confirming the
  document-aware upgrade did not over-trigger on genuine general-knowledge questions.
- **Files:** `app.py` (chatbot rewrite); `src/config.py` (`ROUTE_META`);
  `src/nodes/router.py` (meta category + easy→medium upgrade); `src/nodes/corpus_info.py`
  (new); `src/graph.py` (meta route wiring); `src/ingest.py` (`page_one_chunks`);
  `src/nodes/search.py` (page-1 boost); `tests/test_corpus_info.py` (new),
  `tests/test_router.py`, `tests/test_graph.py`, `tests/test_search.py` (extended).
- **Deviation:** none from the approved spec/plan.
