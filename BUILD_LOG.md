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
