# Adaptive RAG — Comprehensive Change & Architecture Log

> **Purpose of this document.** A self-contained, exhaustive record of everything
> built and changed on this project — the pipeline, the architecture, the refactors,
> and every real problem/bug encountered and how it was fixed. Written to be attached
> to a fresh chat as full context. Read top to bottom; each section stands alone.

---

## 1. What the project is

**Meridian — an Adaptive RAG document Q&A system.** It decides *how much* retrieval and
verification each question needs **before** retrieving, instead of forcing every query
through one fixed search-then-generate pipeline. A router classifies each question and
routes it down one of three modes:

- **Mode 1 — Direct Answer** (`easy`): general-knowledge questions answered from the
  model's own knowledge. No retrieval.
- **Mode 2 — Single-Hop Retrieval** (`medium`): one vector search, then grounded
  generation with inline `[n]` citations (document + page).
- **Mode 3 — Multi-Hop + Self-Critique** (`hard`): decompose the question into
  sub-questions, retrieve evidence for each, generate an answer, then critique every
  claim against the evidence — looping back to search if a claim isn't supported,
  **capped at 3 iterations** so it always terminates.

There is also a fourth, non-LLM path added later — **Corpus Info** (`meta`) — for
questions about the corpus itself ("what documents are loaded?").

**Everything runs locally.** No cloud services, no API keys (after the Groq→Ollama
migration described in §5).

---

## 2. Current architecture (end state)

### Runtime stack
| Layer | Tool | Notes |
|---|---|---|
| Orchestration | LangGraph `StateGraph` | compiled once via `build_graph()` |
| LLM | `qwen2.5:7b` via **Ollama** (local) | was Groq `llama-3.3-70b` originally |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` | local, 384-dim, truncates at 256 tokens |
| Vector store | FAISS `IndexFlatIP` (exact) | disk-persisted, cosine via normalized vectors |
| PDF parsing | PyMuPDF (`fitz`) | 256-token chunks / 50 overlap |
| Frontend | Streamlit chatbot | `st.chat_input` + `st.chat_message` |

### File layout (current)
```
app.py                     Streamlit chatbot UI
src/
  config.py                all knobs: model, chunk size, TOP_K, MAX_ITERATIONS, route labels
  state.py                 RAGState TypedDict — the graph's shared state contract
  ingest.py                PDF parse, heading + title extraction, chunk, embed, FAISS index,
                           search, match_document, page_one_chunks
  graph.py                 LangGraph assembly: router branch, Mode 1/2/3 paths, meta path,
                           critique retry loop
  nodes/
    llm.py                 shared chat() helper — the SINGLE LLM entry point (Ollama)
    router.py              deterministic meta-detection + LLM easy/medium/hard + doc-aware upgrade
    direct_answer.py       Mode 1
    search.py              FAISS retrieval, per-document scoping, page-1 boost, pool+dedup
    generate.py            grounded generation with [n] citations
    decompose.py           Mode 3: split a hard question into sub-questions
    critique.py            Mode 3: verify claims against evidence
    corpus_info.py         meta route: list indexed documents (no LLM call)
tests/                     11 test files (10 offline + 1 live routing harness)
data/documents/            the PDFs (gitignored)
index/                     FAISS index + metadata.json (gitignored, rebuilt by ingest)
docs/superpowers/          design specs + implementation plans for each phase of work
```

### The graph, precisely
```
START → router
router (conditional):
    easy   → direct_answer → END
    medium → search → generate → END
    hard   → decompose → search → generate → critique → (conditional):
                 clean                         → END
                 unsupported & iters < 3       → prepare_retry → search → ... (loop)
                 cap hit                        → END
    meta   → corpus_info → END
```
- **Iteration cap:** `search_node` increments `state["iterations"]` every call; the router
  resets it to 0 per run; the critique conditional edge reads it against
  `config.MAX_ITERATIONS = 3`. No separate counter — this is what guarantees termination.
- **State merge:** plain `TypedDict` overwrite (no `Annotated` reducers) — every node
  returns its complete computed value for each field it touches.

### The RAGState contract (`src/state.py`)
`question, route, mode_label, sub_questions, retrieved, answer, citations,
critique_clean, unsupported_claims, iterations, trace`

### Chunk schema (`src/ingest.py` `Chunk` dataclass)
`chunk_id, text, page_number, section_heading, document_name, document_title`
(`document_title` was added during the general-document work — see §4.)

### Corpus (current)
11 documents, **1695 chunks**: attention_is_all_you_need, bert, roberta, albert,
electra, distilbert, xlnet, t5, gpt2 (the real GPT-2 paper), gpt3, palm.

---

## 3. The build, phase by phase (Phases 0–7)

The project was built in disciplined phases, each with a design spec, an implementation
plan, subagent-driven implementation + review, and live verification. All committed to
`main`.

- **Phase 0/1 — Scaffold + environment.** Directory tree, `requirements.txt`,
  `src/config.py` (locked to `CHUNK_SIZE_TOKENS=256`, `TOP_K=6`, `MAX_ITERATIONS=3`),
  venv with all deps.
- **Phase 2 — Ingestion (`src/ingest.py`).** PyMuPDF parse → chunk → MiniLM embed →
  FAISS `IndexFlatIP` → persist to `index/`. Exposes `load_index()` / `search()`.
- **Phase 3 — Router + Mode 1/2 nodes.** `llm.py` (shared Groq `chat()` at the time),
  `state.py`, `router.py`, `direct_answer.py`, `search.py`, `generate.py`.
- **Phase 4 — Mode 3 nodes.** `decompose.py`, `critique.py`.
- **Phase 5 — Graph assembly (`src/graph.py`).** The core deliverable: wired all nodes
  into a `StateGraph` with the router branch and the critique retry loop. Added a small
  inline `prepare_retry` glue node that feeds critique's `unsupported_claims` back in as
  the next iteration's `sub_questions`.
- **Phase 6 — Streamlit UI (`app.py`).** Verified live in a real browser (via Playwright,
  installed as a one-time dev tool).
- **Phase 7 — Test set + README.** `tests/test_questions.py` (30-question routing
  harness) and a rewritten README. Done later, alongside the production hardening in §7.

---

## 4. Major architectural change #1 — Generalize from research papers to any PDF

**Why.** The project was originally hardcoded to academic papers: the heading detector
only recognized academic section names, the router/decompose prompts named specific RAG
papers, and paths were called `data/papers/`. The requirement was general document
support (any PDF), which PyMuPDF is fully capable of — the limits were hand-tuned
heuristics, not the parser.

**Five changes (spec + 6-task plan):**

1. **Font-size-based heading detection** (replaced keyword regex). A heading is now text
   whose font size is meaningfully larger than the document's detected body-text size —
   domain-agnostic (works for "3.2 Results", "Chapter 4", "SECTION 7", "Amendment XII").
2. **Per-document title extraction** — a new `document_title` field: try embedded PDF
   metadata first, then largest-font text on page 1, then filename stem.
3. **Cross-document retrieval scoping** — `match_document(query)` scores a query's token
   overlap against each document's title + filename; `search()` gained a `document_hint`
   parameter to filter results to one document. Fixes cross-paper contamination (a BERT
   query surfacing RoBERTa chunks) in a corpus of vocabulary-similar papers.
4. **Domain-agnostic prompts** — router/decompose prompts rewritten to be generic
   (removed "Adaptive-RAG, Self-RAG, Chain-of-Verification" references).
5. **Rename** `data/papers/` → `data/documents/`, `PAPERS_DIR` → `DOCS_DIR`.

**Measured result.** Heading-detection accuracy improved from **86% generic/wrong
headings → 18.5%** (a 4.6× reduction), measured directly against the real index.

**Generalization proof.** Ingested the US Constitution (a genuinely non-academic PDF, 19
pages, 117 chunks) end-to-end. Heading detection correctly identified legal structure it
was never designed for ("Article V", "Amendment XII", "SECTION 3"); retrieval was
accurate ("First Amendment" → establishment-of-religion text). Confirms the work isn't
secretly still paper-specific.

---

## 5. Major architectural change #2 — Replace Groq (cloud) with local Qwen via Ollama

**Why.** The Groq free tier's **daily token quota (100,000 tokens/day)** was repeatedly
exhausted during development and live verification — most acutely it blocked a Mode 3
verification for ~72 minutes waiting for the daily reset. The requirement was to remove
the cloud dependency and its rate limit entirely.

**What changed.** Because every LLM call in the project already goes through **one shared
function** (`src/nodes/llm.py` `chat()`), this was a single-file swap — **no node file
changed**:
- `llm.py`: `_client()` returns the `ollama` module (with a reachability check) instead
  of a Groq API-keyed client. `chat()` keeps its exact signature; internally calls
  `ollama.chat(model=..., messages=..., options={temperature, num_predict})`. Removed the
  Groq-specific retry/timeout scaffolding (a local model has no rate limit to retry
  through; failure means Ollama isn't running or the model isn't pulled).
- `config.py`: `GROQ_API_KEY`/`GROQ_MODEL` → `OLLAMA_MODEL = "qwen2.5:7b"`.
- `requirements.txt`: `groq` → `ollama`. `.env.example`: no API key needed.

**Model.** `qwen2.5:7b` (~4.7 GB, Q4) — fits comfortably in 18 GB RAM alongside the
embedding model and FAISS index.

**Known tradeoff (documented, accepted).** Qwen 7B is less reliable than Groq's 70B model
at emitting parseable `[n]` citation markers on **long Mode-3 generations** — the
evidence is retrieved correctly, but citations sometimes fail to resolve. Mode 1/2
citations are reliable. Fixable by moving to a larger local model (e.g. 14B) if needed;
`tests/test_questions.py --mode3` tracks it as a metric.

---

## 6. Major architectural change #3 — Chatbot UI + three query-handling fixes

**Why (UI).** The original UI had a preset-question dropdown that, sitting where the mode
badge appears, repeatedly read as a "mode selector" — users thought they *pick* the
retrieval mode. They don't; the router decides. Reworked into a proper chatbot
(`st.chat_input` + `st.chat_message`, running history), with the routed mode shown as a
small caption per answer ("Answered via single-hop retrieval") reporting the system's own
decision. Flow is now: ingest documents → open UI → type a query → the system routes,
retrieves, and answers on its own.

**Why (three fixes).** Live chatbot testing surfaced four "not working" queries with
three distinct root causes:

1. **"what documents are ingested?"** — no capability existed; it was routed to content
   retrieval and (correctly) found nothing. **Fix A:** added a `meta` route + a
   `corpus_info` node that lists indexed documents (name + title) directly from metadata,
   no LLM call.
2. **"in xlnet who are the authors"** — the author names *are* indexed on page 1, but a
   raw author list ranks low semantically for "who are the authors", so it never reached
   top-k. **Fix B:** `page_one_chunks()` — when a query is confidently scoped to a
   document, its page-1 chunks (title/authors/abstract) are boosted into the candidate
   set. Bounded: only fires on a confident `match_document` hit, so unscoped queries get
   no page-1 noise.
3. **"explain bert"** — answered from the model's own memory (route `easy`) instead of
   the BERT document. **Fix C:** when the router labels a question `easy` AND
   `match_document` finds it names an indexed document, upgrade the route to `medium`
   (answer from the document with citations). Strictly gated to `easy` — meta/medium/hard
   are never altered.

**Verified.** All four queries fixed end-to-end; routing accuracy held at 93.3%.

---

## 7. Production hardening (done alongside Phase 7)

- **GPT corpus data-accuracy fix.** The file named `gpt2.pdf` actually contained the
  **GPT-3** paper ("Language Models are Few-Shot Learners") — an arXiv-ID mixup from the
  original download. Renamed it to `gpt3.pdf` and added the real GPT-2 paper. Also made
  `match_document` de-hyphenate model names so "GPT-2" produces a `gpt2` token matching
  the filename stem (previously gpt2/gpt3 tied on their near-identical titles and fell
  back to no-match).
- **Closed a test-coverage gap + corrected an overclaim.** An early BUILD_LOG entry
  claimed router/search/generate were "unit-tested", but no committed test files existed
  for router or generate. Added `tests/test_router.py` and `tests/test_generate.py` and
  corrected the record honestly.
- **Junk-metadata title rejection.** Found (via the Constitution test) that some PDFs
  have junk embedded titles (e.g. "constitution_pdf2"). Added `_looks_like_filename()` so
  junk/filename-shaped metadata is rejected in favor of font-based extraction.
- **README rewrite** to reflect reality (Ollama setup, general documents, correct stack).

---

## 8. Every real problem/bug encountered — and the fix

This is the honest ledger of things that actually went wrong, most of which were caught
by verifying against the **real corpus** rather than trusting synthetic tests.

| # | Problem | Root cause | Fix |
|---|---|---|---|
| 1 | **Heading detection near-useless** — initial font-size threshold detected ZERO real section headings on the BERT paper | `_HEADING_SIZE_RATIO = 1.15` was too strict; real academic headings are only ~10% larger than body text, not 15%+. Synthetic test PDFs had exaggerated ratios (1.6–2.0×) that hid this | Recalibrated to **1.08** after sweeping all 10 real papers. BERT went from 2 → 14 real headings. Added a realistic-ratio regression test |
| 2 | **arXiv watermark picked as document title** | Title extraction took the single largest-font span on page 1; arXiv's rotated sidebar watermark is large-font | Filter to horizontal-only text lines (`line.get("dir")`) |
| 3 | **Small-caps titles truncated** — "ALBERT" extracted as just "A" | Small-caps titles split one word across two font sizes (17pt "A" + 13.8pt "LBERT"); taking the single largest span grabbed only the leading letters | Group spans into lines and use a size-tolerance threshold (0.85) instead of a single largest span |
| 4 | **match_document scoring mathematically unsatisfiable** — two of the plan's own test fixtures required opposite outcomes from identical scores | Plain unweighted token overlap can't distinguish a single-word incidental match from a real match | Added a stopword filter + `_STEM_MATCH_WEIGHT` (filename-stem matches count more) |
| 5 | **Groq daily quota exhaustion** blocked verification for ~72 min | Free-tier 100k tokens/day; Mode 3 makes up to 8 LLM calls per question | Migrated the whole LLM layer to local Ollama/Qwen (§5) |
| 6 | **`gpt2.pdf` contained the GPT-3 paper** | arXiv-ID mixup at download time (2005.14165 is GPT-3, not GPT-2) | Renamed to gpt3.pdf, downloaded the real GPT-2 paper, re-ingested; de-hyphenated model-name matching |
| 7 | **Junk embedded PDF title trusted** — Constitution titled "constitution_pdf2" | Metadata-first title strategy trusted a non-empty but garbage title | `_looks_like_filename()` rejects filename-shaped metadata, falls back to font-based extraction |
| 8 | **UI looked like it required mode selection** | A preset-question dropdown sat where the mode badge appears | Reworked into a chatbot; mode shown as an output caption, not a control |
| 9 | **"who are the authors" returned nothing** | Author block on page 1 ranks low semantically | Page-1 chunk boosting for document-scoped queries (Fix B, §6) |
| 10 | **"explain bert" answered from model memory** | Router classified it `easy`, so no retrieval | Document-aware `easy → medium` upgrade (Fix C, §6) |
| 11 | **`meta` route MASSIVELY over-triggered** — nearly every question ("Hi", "who are the authors in xlnet") returned the document list instead of an answer | The 7B model cannot reliably do a 4-way easy/medium/hard/meta classification; it over-applied `meta` | **Removed `meta` from the LLM's job.** Detect meta questions **deterministically** with a tight regex (`is_meta_question`) that runs BEFORE the LLM router: requires a collection noun + a meta verb/list-frame AND no named document. The LLM router went back to a clean, reliable 3-way classification |
| 12 | **em-dash in user-visible output** (corpus list) | `corpus_info` joined name/title with " — " | Replaced with ": " (project constraint: no em-dashes in user-visible strings) |

**Recurring lesson:** the small local model is unreliable at fine-grained classification,
and synthetic tests hid real bugs. The fix pattern throughout was (a) move crisp,
rule-shaped decisions out of the LLM into deterministic code, and (b) always verify
against the real corpus, not just fixtures.

---

## 8b. Conceptual issues, questions raised, and decisions discussed

Beyond the code bugs in §8, a number of issues were raised and discussed that were about
*understanding, scope, or design* rather than a single broken line. Recording them for
completeness, since they shaped the work.

### Conceptual confusions raised (and clarified)

- **"Why would the user select what mode to go with?" / "the queries aren't working"
  (mode selection).** Raised twice. The dropdown was *not* a mode selector — it was a
  preset-question picker — but placed where the mode badge appears it read as one. The
  underlying truth: **the user never picks the mode; the router decides it automatically
  before retrieval.** First addressed by explanation, then (when raised again with the
  desired flow: ingest → open UI → type query → system does everything) fixed properly by
  reworking the UI into a chatbot (§6). This is the same confusion that also surfaced as
  the `meta` over-triggering bug (§8 #11) — the model, like the UI, was making a
  selection the user didn't want it to.

- **"I want the model to efficiently pick up and choose the retrieval strategy by
  itself."** Clarified that the router *already* does this entirely on its own — the
  concern was really about *how well* it does it, which is what motivated actually
  measuring it (the Phase-7 30-question harness → 93.3%). The routing was real and working;
  it just had never been quantified.

- **"Why is this project only handling research papers... can't make this specific to
  research papers."** A correct challenge to the project's scope. Led to the general-
  document architectural change (§4). The honest finding: the architecture was never
  paper-specific (PyMuPDF, chunking, FAISS, the graph are all document-agnostic) — only a
  handful of *heuristics and prompts* were hand-tuned for papers. Those were generalized.

- **"Shouldn't we have a database to make it more permanent and efficient?"** Discussed,
  concluded **no change needed** for the current scope. The FAISS + JSON-sidecar setup is
  already persistent (survives restarts, no re-embedding). A real vector DB would solve a
  *scaling/concurrency* problem the project doesn't have yet (measured: FAISS flat search
  is ~0.036 ms at ~1500 chunks; fine into the hundreds of thousands). It would matter only
  if the project became multi-user, continuously-growing, or a shared service. Flagged as
  a "revisit if scope changes" item, not acted on.

- **"Where are the chunks stored after ingestion?"** Clarification, not a bug: chunks live
  in two files under `index/` — `faiss.index` (the 384-dim vectors) and `metadata.json`
  (the chunk text + metadata, aligned by position). Both gitignored, rebuilt by
  `python -m src.ingest`.

### Scope / sequencing decisions made during the chat

- **Production-readiness audit.** The project was called "so-called production ready" with
  skepticism, prompting a full critical audit (after expanding the corpus to 10 papers).
  The audit produced a prioritized list of 5 gaps; they were then worked through:
  1. LLM retry/rate-limit fragility → first widened Groq's retry budget, then (when the
     quota was hit again) replaced Groq with local Ollama entirely (§5).
  2. Cross-paper retrieval contamination → the `match_document` + `document_hint` scoping
     (§4).
  3. Heading-detection quality (measured at 86% wrong) → font-size detection (§4, and bug
     §8 #1).
  4. Router/decompose prompt-corpus mismatch → domain-agnostic prompts (§4).
  5. A live end-to-end pass on the expanded corpus + the Phase-7 test set → done (§7, §9).
  - The audit also **caught a documentation overclaim**: an earlier log said
    router/search/generate were unit-tested when no such committed tests existed (fixed,
    §7).

- **Corpus scaled 2 → 10 → 11 papers.** Deliberately expanded to stress-test "production
  readiness" against a larger, vocabulary-similar corpus (which is what *exposed* the
  cross-paper contamination and the FAISS-scale question). Later became 11 when the GPT-2
  paper was added to correct the mislabeling (§8 #6).

- **The cross-paper contamination fix was designed, paused, then folded into the general-
  document work.** Its original design assumed paper-shaped PDFs (title extraction), so
  rather than build it twice, it was merged into the general-document generalization —
  avoiding immediate rework.

- **Groq → local model: full replacement, not a fallback.** Explicitly chose to remove
  Groq entirely (no dual-path config toggle) rather than keep it as a fallback, because
  the goal was to eliminate the rate-limit problem, not manage around it.

- **Qwen 7B citation limitation: accept and document, don't over-engineer.** When Mode-3
  citation formatting proved unreliable on the smaller model, the decision was to record
  it honestly as a known tradeoff (with a metric to track it) rather than prompt-engineer
  around a fundamental small-model limitation. A 14B model is the named escape hatch.

### External constraints hit (not code issues, but they shaped the session)

- **Groq daily token quota** was hit repeatedly (100k tokens/day). It blocked live
  verification more than once and was the direct trigger for the Ollama migration.
- **No browser-automation tool was pre-installed**, so Playwright + headless Chromium were
  installed as one-time dev/verification tools (never added to `requirements.txt`) to
  drive the Streamlit UI in a real browser. `pytest` was similarly installed dev-only for
  the monkeypatch-based tests.
- **Sandbox has no network access to huggingface.co**, producing harmless retry warnings
  during embedding-model loads (the model loads from local cache regardless). Cosmetic
  noise, not a failure.

### Items raised but intentionally NOT changed (with rationale)

- A stray trailing-newline change in `direct_answer.py` predating the session — committed
  as a tidy, not reverted.
- The `INTERVIEW_PREP.md` personal-notes file — committed as-is (user's own working
  material), not rewritten despite being partly stale.
- The `match_document` being called twice per medium query (once in the router for the
  upgrade decision, once in search for scoping) — reviewed and confirmed **not** a
  correctness issue (both read the lru-cached index, pure functions, no inconsistency
  risk within a run); left as-is rather than adding coupling to thread it through state.
- Unbounded retrieval-pool growth feeding the generate prompt — acknowledged as an
  inherent, bounded consequence of the page-1 boost; acceptable for a local demo, flagged
  as an optional future hardening only if the corpus scales.

---

## 9. Testing

- **10 offline unit-test files** (no LLM/network needed — pure parsing/branching logic):
  `test_router`, `test_generate`, `test_decompose`, `test_critique`, `test_graph`,
  `test_search`, `test_corpus_info`, `test_ingest_headings`, `test_ingest_title`,
  `test_ingest_match_document`. **67 tests, all passing.**
- **1 live harness** — `tests/test_questions.py`: 30 labelled questions (10 easy / 10
  medium / 10 hard) through the live router, reports routing accuracy; `--mode3` also runs
  hard questions through the full graph and measures citation resolution.
  - **Current routing accuracy: 28/30 = 93.3%** (easy 10/10, medium 8/10, hard 10/10).
    The two medium misses route to *hard*, which fails safe (more retrieval, not less).

Run offline: `for t in tests/test_*.py; do [ "$t" = "tests/test_questions.py" ] && continue; python "$t"; done`
(or `pytest tests/ --ignore=tests/test_questions.py`). Run the harness:
`python -m tests.test_questions [--mode3]`.

---

## 10. How to run it

```bash
# One-time: install Ollama (https://ollama.com), then:
ollama pull qwen2.5:7b

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Drop PDFs into data/documents/, then build the index:
python -m src.ingest

# Launch the chatbot:
streamlit run app.py
```
No API key or `.env` needed. Ollama must be running (background service on macOS, else
`ollama serve`).

---

## 11. Known limitations (honest, current)

1. **Qwen-7B Mode-3 citation formatting** — long multi-hop answers sometimes omit
   parseable `[n]` markers, so citations don't resolve even though evidence was retrieved
   correctly. Mode 1/2 are reliable. Larger local model would improve it.
2. **Meta detection is pattern-based, not exhaustive** — a badly typo'd corpus question
   ("explan the documents") can miss the pattern and fall through to the LLM router
   (routes to retrieval — a safe failure, not the document list).
3. **Cross-document scoping is conservative** — ambiguous queries search the whole corpus
   rather than risk scoping to the wrong document.
4. **Scope** — no AWS, no auth, no multi-user, no cross-question conversational memory
   (chat history is visual only; each question is routed independently).

---

## 12. Working conventions used throughout

- Every non-trivial change went **design spec → implementation plan → subagent-driven
  implementation + independent review → live verification**, with specs/plans committed
  under `docs/superpowers/`.
- `PLAN.md` and `BUILD_LOG.md` are **append-only** history — never rewritten; corrections
  are added as new entries.
- `BUILD_LOG.md` has 16 numbered entries covering the whole arc; this document
  summarizes them but the BUILD_LOG has the fine-grained per-step detail.
- No emoji / no em-dashes in user-visible strings (a deliberate stylistic constraint).
