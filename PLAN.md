# Execution Plan — Adaptive RAG (Local Demo, No AWS)

> **Status of this document:** Phases 0–3 complete (see BUILD_LOG.md for details).
> Execution happens **phase by phase**, and each phase pauses for your go-ahead before
> the next one starts. This is the single source of truth for *what* gets built and
> *in what order*.

---

## 0. Architecture confirmation (does this plan match the finalized spec?)

**Yes.** This plan implements the finalized architecture without simplifying the core:

| Spec requirement | Plan honors it? |
|---|---|
| 3 distinct modes (No-Retrieval / Single-Hop / Multi-Hop+Critique) | ✅ Not collapsed |
| Router classifies `easy` / `medium` / `hard` **before** retrieval | ✅ Phase 3 |
| LangGraph graph exactly as diagrammed | ✅ Phase 5 |
| Conditional edge critique → search, **cap 3 total iterations** | ✅ Phase 5 (the headline part) |
| Groq for all 4 LLM steps, `GROQ_API_KEY` from `.env` | ✅ |
| Embeddings: local sentence-transformers `all-MiniLM-L6-v2` | ✅ |
| Vector store: FAISS, persisted to disk (no re-embed each run) | ✅ Phase 2 |
| PDF parsing: PyMuPDF | ✅ Phase 2 |
| Frontend: Streamlit, calls graph directly (no FastAPI/API GW) | ✅ Phase 6 |
| Chunking 512 tokens / 50 overlap | ✅ (see Deviation D1) |
| Per-chunk metadata: chunk_id, page_number, section_heading, document_name | ✅ Phase 2 |
| File structure as specified | ✅ (see Deviation D2) |
| `BUILD_LOG.md` maintained throughout | ✅ every phase |
| Modes 1 & 2 prioritized over Mode 3 if time short | ✅ Phase order + Phase 7 |
| No AWS, no auth, no multi-user, no extra features | ✅ |

### Deviations from a literal reading (both minor, both flagged for approval)

- **D1 — Chunk size vs. embedding window. [RESOLVED → 256].** `all-MiniLM-L6-v2`
  truncates at 256 tokens; the spec asked for 512. **Decision (with user): use 256/50**
  so the embedding represents the whole chunk (no silent truncation), and bump `TOP_K`
  4 → 6 to offset the smaller chunks. This is the only functional deviation from spec.
- **D2 — Retrieval helper location.** The search node needs a shared "load FAISS +
  embed query + top-k search" function. The spec's file list has no `retriever.py`, so
  this helper lives **inside `ingest.py`** and is imported by `src/nodes/search.py`.
  Keeps the file list exactly as specified (no new files).

---

## Phase status legend
- ☐ not started ☑ done ◐ in progress

## Phase 0 — Scaffold  ☑ (already completed before this plan)
Created before this plan was drafted; listed for completeness.
- **Deliverables:** directory tree (`data/papers/`, `src/`, `src/nodes/`, `tests/`),
  `requirements.txt`, `.env.example`, `.gitignore`, `src/config.py`, `BUILD_LOG.md`,
  git repo initialized.
- **Exit check:** files exist; `config.py` holds every knob (models, `MAX_ITERATIONS=3`,
  chunk size, top-k, paths, mode labels).

## Phase 1 — Environment  ☑ (already completed before this plan)
- **Deliverables:** `venv/` (Python 3.13.7), all of `requirements.txt` installed.
- **Exit check:** `import fitz, faiss, sentence_transformers, langgraph, groq, streamlit`
  all succeed. (Already verified.)

> **Everything below this line is NOT yet built. Awaiting your go-ahead per phase.**

---

## Phase 2 — Ingestion (`src/ingest.py`)  ☑
Build the offline corpus pipeline and prove it standalone before any graph work.

- **What it does:**
  1. Walk `./data/papers/*.pdf`, parse each with **PyMuPDF** (page text + best-effort
     section heading detection via font-size / heading heuristics).
  2. Chunk to **256 tokens / 50 overlap** (see Deviation D1).
  3. Embed locally with `all-MiniLM-L6-v2`.
  4. Build a **FAISS** index; persist to `index/faiss.index` + `index/metadata.json`
     (chunk text + metadata sidecar), so subsequent runs skip re-embedding.
  5. Expose `load_index()` and `search(query, k)` helpers (per Deviation D2) for the
     search node to import.
- **Metadata per chunk:** `chunk_id`, `page_number`, `section_heading`, `document_name`.
- **Standalone test (no Groq needed):** generate a small **synthetic PDF** in the
  scratchpad, run ingestion over it, confirm index builds, persists, reloads, and a
  sample query returns sensible top-k chunks with correct metadata.
- **Exit check:** `python -m src.ingest --self-test` builds + persists an index;
  reload + search works. **PASSED** — 3 chunks, all headings detected, max chunk 60
  tokens (≤256 window), reload+search returns correct top hit (score 0.586).
- **Log:** BUILD_LOG entry 5 (heading heuristic, offset-based chunking bug fix,
  synthetic-PDF textbox-wrap bug fix).
- **Status:** done, including real-PDF ingestion. `data/papers/` now holds two real
  papers ("Attention Is All You Need", BERT) fetched from arXiv; `python -m src.ingest`
  indexed 132 chunks and a real query ("What is self-attention and how is it computed?")
  returned correct, relevant top-k hits with proper document/page metadata. **Known
  weakness:** the heading heuristic mislabels most sections as "Abstract" on real
  academic PDF layouts (worked fine on the synthetic 3-section test) — heading metadata
  is unreliable for now, text/embedding/retrieval quality is not affected.

## Phase 3 — Nodes, part A: routing + the two simple paths  ☑
Build the nodes that make **Mode 1 and Mode 2** work (spec's priority).

- `src/nodes/llm.py` — shared Groq client/`chat()` helper for all nodes (Deviation:
  not in original spec's file list, thin shared helper).
- `src/state.py` — shared `RAGState` TypedDict contract (Deviation: same as above).
- `src/nodes/router.py` — Groq call, classify `easy|medium|hard`, robust parse + fallback
  (fails safe to "medium"/retrieve on parse failure).
- `src/nodes/direct_answer.py` — Mode 1: answer directly, **zero** vector search.
- `src/nodes/search.py` — FAISS top-k via the `ingest.py` helper; writes chunks to state;
  pools + dedups by chunk_id; bumps iteration counter for the Phase-5 conditional edge.
- `src/nodes/generate.py` — answer from retrieved chunks **with citations**
  (document_name + page), resolved from `[n]` markers in the answer text.
- **Isolation test:** all non-LLM paths unit-tested offline (router parsing/fallback,
  search pooling/dedup/iteration bump, generate citation extraction). All **PASSED**.
  `llm.chat()` confirmed to raise `LLMConfigError` when key is absent.
- **Exit check:** router returns a valid label; direct_answer/search/generate each produce
  correct output shapes in isolation. Met for all non-LLM paths.
- **Log:** BUILD_LOG entry 6 (prompt design, parse-failure fallback behavior); entry 8
  (live Groq routing spot-check).
- **Live Groq routing — done.** `GROQ_API_KEY` added to `.env` (gitignored). Spot-tested
  `route_question()` against 3 real questions spanning all three labels — all classified
  correctly: "What is the capital of France?" → `easy`, "What does the Transformer paper
  say about multi-head attention?" → `medium`, "Compare how BERT and the Transformer
  paper each handle positional information, and explain any tradeoffs." → `hard`. Full
  30-question accuracy sweep still deferred to Phase 7.

## Phase 4 — Nodes, part B: Mode 3 building blocks  ☑
- `src/nodes/decompose.py` — Groq splits a hard question into sub-questions (2-3,
  newline-parsed, capped at 3, fails safe to `[question]` on unparseable output).
- `src/nodes/critique.py` — checks **every claim** in the draft answer against the
  retrieved evidence; returns `clean` vs `unsupported` + which claims failed (the signal
  the conditional edge reads). Sentinel-line response format (`VERDICT:`/`CLAIMS:`),
  fails safe to `clean` on unparseable output. Unsupported claims are phrased as
  search-friendly questions (not verbatim quotes) so Phase 5 can feed them back into
  `search_node` as the next iteration's queries.
- **Isolation test — done.** Offline parsing unit tests for both nodes (newline
  splitting/cap/fallback for decompose; sentinel-line parsing across 5 branches for
  critique) all **PASSED**, no Groq needed.
- **Live spot-check — done.** `decompose()` on a real hard question returned 3 clean,
  independently-searchable sub-questions. `critique()` correctly flagged a deliberately
  fabricated claim ("trained on 8 V100 GPUs for 12 days") while leaving a grounded claim
  (Adam optimizer) unflagged, and returned clean on a fully-grounded answer.
- **Exit check:** met — both nodes produce the state fields (`sub_questions`,
  `critique_clean`, `unsupported_claims`) the Phase-5 conditional edge will branch on.
- **Log:** BUILD_LOG entry 9.

## Phase 5 — Assemble the graph (`src/graph.py`)  ☑  *(the core deliverable)*
- Shared **state** (`src/state.py`'s `RAGState`, unchanged from Phase 3) is used as-is
  with LangGraph's default overwrite merge semantics (no `Annotated` reducers needed —
  every node already returns its complete computed value per field).
- Nodes wired into a `StateGraph` exactly as planned:
  - router → conditional branch to `direct_answer` | `search` | `decompose`.
  - Mode 2: search → generate → END.
  - Mode 3: decompose → search → generate → critique →
    **conditional edge**: `clean` → END; `unsupported` **and** `iterations < 3` → a
    small inline `prepare_retry` glue node (sets `sub_questions = unsupported_claims`)
    → back to search; else (cap hit) → END.
- The **3-iteration cap** enforced via `config.MAX_ITERATIONS` read in the conditional
  edge function — no new counter; `search_node` already increments `iterations` every
  call (Phase 3, unchanged).
- **Live dry-run results — done, all three modes verified:**
  - **Mode 1** ("What is a vector database?"): routed `easy`, answered directly,
    `citations: []`, trace = `router → direct_answer` only.
  - **Mode 2** ("According to the BERT paper, what pretraining tasks does BERT use?"):
    routed `medium`, correctly identified MLM + NSP with a citation into `bert.pdf`
    p8, `iterations: 1`, trace = `router → search → generate` only.
  - **Mode 3** ("Compare how BERT and the Transformer paper each handle positional
    information..."): routed `hard`, looped through `decompose → search → generate →
    critique` then `prepare_retry → search → generate → critique` **twice more**,
    hit the cap at exactly `iterations: 3` (never more — the trace shows exactly 3
    `search` entries), terminated with `critique_clean: False` (expected cap-hit
    outcome, not a bug) and a 6-citation answer spanning both papers.
- **Exit check:** met — graph compiles, all three modes run correctly, loop provably
  terminated at exactly 3 passes on a genuinely hard question.
- **Log:** BUILD_LOG entry 10.

## Phase 6 — Streamlit UI (`app.py`)  ☐
- Text input + **Run** button; calls the compiled graph directly (no API layer).
- Output shows: the **answer**, **which mode fired** (visually prominent — the headline
  of the demo), and **citations** (document + page). For Mode 3, show iteration count.
- **Exit check:** `streamlit run app.py` launches; a question flows through and the mode
  badge is unmistakable.
- **Log:** BUILD_LOG entry.
- **⏸ PAUSE for your review before Phase 7.**

## Phase 7 — Test set + README  ☐
- `tests/test_questions.py` — runnable **30-question set (10 easy / 10 medium / 10 hard)**;
  prints each question, the routed mode, and pass/fail vs expected route so you can
  validate routing accuracy.
- `README.md` — setup: Groq key into `.env`, `pip install -r requirements.txt`, run
  ingestion, launch Streamlit; note the FAISS persistence + no-AWS scope.
- Final **BUILD_LOG** entry: honest Mode 3 status (stable / needs debugging / cut), per
  the spec's "priority if time runs short" clause.
- **Exit check:** test script runs and reports routing results; README reproduces setup
  from scratch.
- **⏸ FINAL review.**

---

## Cross-cutting rules (every phase)
- **No AWS. No FastAPI/API Gateway.** Streamlit → graph directly.
- **Don't simplify** the three modes or the conditional loop.
- **No unlisted features** (no auth, multi-user, or persistence beyond the FAISS index).
- **BUILD_LOG.md** gets an entry per meaningful step, factual and brief.
- Each phase is independently testable; we pause between phases for your review.

## Open questions for you (before Phase 2 starts)
1. **D1:** keep 512-token chunks (embedding sees leading ~256), or switch to 256 so the
   embedding sees the whole chunk? *Default: keep 512 per spec.*
2. **Groq key + PDFs:** confirmed earlier you'll add `GROQ_API_KEY` and drop real PDFs in
   `data/papers/` yourself; phases needing live Groq will be tested once those exist.
   Until then I test with a synthetic PDF and defer live routing checks.
