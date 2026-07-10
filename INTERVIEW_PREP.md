# Adaptive RAG — Interview Prep & Technical Deep-Dive

> Everything built so far (Phases 0–3), the pipeline, the technical rationale, and the
> hard questions an interviewer will ask — with answers. Read top-to-bottom once; use the
> "Cross-Questioning" sections to rehearse.

---

## 1. One-sentence pitch

> "It's a document Q&A system that decides **how** to retrieve *before* it retrieves —
> routing each question into one of three modes by complexity, instead of forcing every
> query through the same fixed search-then-generate pipeline."

**Why it matters:** naive RAG retrieves for *every* question, wasting latency/tokens on
questions the model already knows, and under-retrieving on complex multi-hop ones. This
system adapts the retrieval strategy to the question.

**Grounded in three papers:**
| Paper | Venue | What we borrow |
|---|---|---|
| Adaptive-RAG (Jeong et al.) | NAACL 2024 | Classify query complexity *before* retrieval |
| Self-RAG (Asai et al.) | ICLR 2024 | Reflect on whether retrieval is needed; critique own output |
| Chain-of-Verification (Dhuliawala et al.) | ACL 2024 | Verify every claim against evidence before returning |

---

## 2. The three modes (the core of the project)

| Mode | Trigger | Pipeline | Retrieval |
|---|---|---|---|
| **1 — No Retrieval** | `easy` | LLM answers directly | **zero** searches |
| **2 — Single-Hop** | `medium` | 1 FAISS search → top-k → generate w/ citations | 1 search |
| **3 — Multi-Hop + Self-Critique** | `hard` | decompose → search per sub-q → generate → critique → loop | 1–3 searches |

**Mode 3 loop:** decompose into sub-questions → search each → chain evidence → generate →
critique every claim vs. evidence → if unsupported claims, **loop back to search** →
**hard cap 3 iterations** → exit when clean or cap hit.

---

## 3. Architecture / pipeline (LangGraph)

```
Question
   │
   ▼
ROUTER  (LLM: easy | medium | hard)          ← Adaptive-RAG idea
   │
   ├── easy ───►  DIRECT_ANSWER ─────────────────────────► END   (Mode 1)
   │
   ├── medium ─►  SEARCH ─► GENERATE ────────────────────► END   (Mode 2)
   │
   └── hard ───►  DECOMPOSE ─► SEARCH ─► GENERATE ─► CRITIQUE     (Mode 3)
                                 ▲                        │
                                 │                        ▼
                                 │              ┌── clean ──► END
                                 └── unsupported & iterations < 3 ┘
                                    (else: cap hit ──► END)
```

- **Nodes are plain Python functions** that take `RAGState` and return the fields they change.
- **Edges** connect nodes; LangGraph merges each node's returned dict into shared state.
- The **conditional edge after CRITIQUE** is what creates the loop — the single most
  important thing to be able to explain (see §7).

---

## 4. Tech stack & *why each choice* (be ready to defend every row)

| Layer | Tool | Why this, why not alternatives |
|---|---|---|
| Orchestration | **LangGraph** | Native support for **cyclic** graphs + conditional edges. Plain LangChain chains are DAGs — they can't express the critique→search loop cleanly. |
| LLM (route/decompose/generate/critique) | **Groq** (`llama-3.3-70b-versatile`) | Very low latency inference; free tier; one capable model for all 4 steps. Steps differ by **prompt**, not model. |
| Embeddings | **sentence-transformers `all-MiniLM-L6-v2`** | Fully local (no API cost, no data leaves the machine), 384-dim, fast on CPU. |
| Vector store | **FAISS** (`IndexFlatIP`, in-memory, persisted) | Exact search, zero infra, trivial to persist. Flat is fine for a handful of papers. |
| PDF parsing | **PyMuPDF (fitz)** | Fast, accurate text + layout blocks; lets us do best-effort heading detection. |
| Frontend | **Streamlit** | Calls the graph **directly** — no API layer needed for a demo. |
| Chunking | **256 tokens / 50 overlap** | See §6 — this is a deliberate, defensible deviation. |

**Scope guardrail (say this if asked about production):** AWS is explicitly *out of scope*
for this build — it's a documented future deployment target. No S3/Lambda/Bedrock/RDS/
API-Gateway/FastAPI. Streamlit → graph directly. This keeps the demo fully local and
runnable.

---

## 5. What's actually built (Phases 0–3)

| Phase | Deliverable | Status |
|---|---|---|
| 0 | Scaffold, `requirements.txt`, `config.py`, `.env.example`, `.gitignore`, `BUILD_LOG.md` | ✅ done |
| 1 | `venv` + all deps installed & import-verified | ✅ done |
| 2 | `src/ingest.py` — parse→chunk→embed→FAISS→persist + `search()`/`load_index()` | ✅ done, self-test PASSED |
| 3 | `router`, `direct_answer`, `search`, `generate` nodes + `llm.py` + `state.py` | ✅ done, unit tests PASSED |
| 4 | `decompose`, `critique` nodes | ⏳ next |
| 5 | `graph.py` — wire nodes + conditional loop + 3-iter cap | ⏳ |
| 6 | `app.py` — Streamlit UI (prominent mode badge + citations) | ⏳ |
| 7 | `tests/test_questions.py` (30 Qs) + `README.md` | ⏳ |

**File map (current):**
```
src/
├── config.py          all knobs: models, chunk size, TOP_K, MAX_ITERATIONS, paths, labels
├── state.py           RAGState TypedDict — the shared contract every node reads/writes
├── ingest.py          offline pipeline + runtime search()/load_index()
└── nodes/
    ├── llm.py         shared Groq client + chat() + LLMConfigError
    ├── router.py      classify easy/medium/hard (defensive parse + fail-safe)
    ├── direct_answer.py   Mode 1: answer, no retrieval
    ├── search.py      FAISS top-k; pools+dedups; bumps iteration counter
    └── generate.py    cited answer; [n] markers → structured citations
```

---

## 6. The ingestion pipeline in detail (`ingest.py`)

**Flow:** `PDF → parse (PyMuPDF) → section-tag → chunk (256/50) → embed (MiniLM) →
FAISS IndexFlatIP → persist (faiss.index + metadata.json)`

**Per-chunk metadata:** `chunk_id`, `page_number`, `section_heading` (best-effort),
`document_name` — required so answers can cite exact sources.

**Key engineering decisions (each is a likely question):**

1. **Token counting uses the embedder's own tokenizer.** "256 tokens" means exactly the
   256-token window MiniLM sees — chunk boundaries line up with the embedding window.
2. **Chunk by *character offsets*, not by decoding token IDs.** MiniLM's WordPiece
   tokenizer lowercases and mangles punctuation on decode (`self-critique` →
   `self - critique`). Decoding would corrupt stored text → bad citations + bad LLM
   context. We use `return_offsets_mapping` to slice the **original string** verbatim.
3. **Normalized embeddings + `IndexFlatIP`** → inner product **equals cosine similarity**.
   Simpler and exact vs. building an L2 index and converting.
4. **Persistence** = FAISS index file + JSON metadata sidecar → **no re-embedding** on
   every run. Load once, cached for the process (`lru_cache`).
5. **Best-effort headings** via `get_text("blocks")` + regex (numbered headings + common
   section keywords like Introduction/Method/Results). Font-size analysis would be more
   precise but heavier — "best-effort" per spec.

**Two real bugs caught during the build (good stories to tell):**
- *Lossy chunk text* — decoded token IDs → lowercased text. Fixed via offset slicing.
- *Truncated synthetic test PDF* — `insert_text` at a fixed point ran off-page and
  PyMuPDF only extracted the visible part. Fixed with `insert_textbox` (wraps in a rect).
  Note: a **test-fixture** flaw, not a chunking bug — real papers lay out text properly.

---

## 7. The self-critique loop — explain this cold (highest-value topic)

This is "the part of LangGraph I most want working correctly and able to explain."

**Mechanics:**
1. `CRITIQUE` checks **every claim** in the draft answer against retrieved evidence
   (Chain-of-Verification idea). It sets `critique_clean` (bool) + `unsupported_claims`.
2. A **conditional edge** function reads state and returns the next node:
   - `critique_clean == True` → **END** (answer is grounded).
   - `unsupported_claims` present **AND** `iterations < MAX_ITERATIONS (3)` → back to
     **SEARCH** (retrieve more evidence, regenerate, re-critique).
   - else (**cap hit**) → **END** (return best-effort answer + note the limitation).
3. The `iterations` counter is **incremented in the search node**, so every loop pass
   provably advances it. That guarantees termination — the loop *cannot* run forever.

**Why the cap lives in the counter, not a timer:** deterministic, testable, explainable.
`iterations < 3` is a pure function of state → the graph is easy to reason about and unit
-test (we can assert "runs ≤ 3 passes then exits" without any LLM).

**Why loop back to SEARCH (not GENERATE):** unsupported claims mean we lack *evidence*,
not that we phrased poorly. More retrieval is the fix; the search node **pools** new hits
with existing ones so the evidence set grows across iterations.

---

## 8. State design (`RAGState`)

A single `TypedDict` every node reads/writes. Nodes return only changed keys; LangGraph
merges them. Key fields:

- `question` → input
- `route`, `mode_label` → routing decision + UI label
- `sub_questions` → Mode 3 decomposition
- `retrieved` → pooled chunks (accumulated across search passes, deduped by `chunk_id`)
- `answer`, `citations` → output (citations = the subset actually cited, with doc+page)
- `critique_clean`, `unsupported_claims` → the loop's branching signal
- `iterations` → the cap counter
- `trace` → ordered log of which nodes fired (for the UI / debugging)

**Why a shared typed contract:** field names never drift between node files; every node
type-checks against one source of truth.

---

## 9. Design decisions & deviations from spec (own them proactively)

| # | Decision | Rationale | Spec deviation? |
|---|---|---|---|
| D1 | **256-token chunks** (spec said 512) | MiniLM truncates at 256; a 512 chunk would be **half-embedded** (silent truncation). 256 → embedding represents the whole chunk. Bumped `TOP_K` 4→6 to offset smaller chunks. | **Yes**, deliberate & discussed |
| D2 | `search()`/`load_index()` live **in `ingest.py`** (no `retriever.py`) | Keeps the spec's file list exact; ingestion and retrieval share the same model/paths. | Minor (no new file) |
| — | Added `llm.py` + `state.py` | Thin shared helpers: DRY LLM calls + single-sourced state. Not features. | Minor additions |
| — | One model for all 4 LLM steps | Steps differ by **prompt**, not capability. Cheaper, simpler to explain. | No |
| — | `temperature = 0.0` | Routing/critique are **classification** tasks — want stable, reproducible decisions. | No |
| — | Router **fails safe to "medium"** | On unparseable LLM reply, a wrong retrieve is cheaper than a confident ungrounded answer. | No |

---

## 10. Testing done so far (evidence, not claims)

- **Ingestion self-test** (`python -m src.ingest --self-test`): builds a synthetic
  3-section PDF, runs parse→chunk→embed→persist→reload→search, asserts the correct
  section is the top hit. **PASSED.**
- **Phase-3 unit tests (offline, no Groq key):** router parsing + fallback; search
  single/multi-hop pooling/dedup/iteration bump; generate `[n]`→citations + no-evidence
  path; missing-key raises a clear `LLMConfigError`. **ALL PASSED.**
- **Not yet tested:** live routing *accuracy* — needs `GROQ_API_KEY` in `.env`; validated
  in Phase 7 with the 30-question set (10 easy / 10 medium / 10 hard).

**Testing philosophy to state:** the deterministic logic (parsing, merging, counting,
citation resolution, loop termination) is unit-tested **without the LLM** by mocking the
`chat()` boundary. Only the model's *judgment* is left to live validation — everything
around it is proven offline.

---

## 11. Cross-questioning — likely interview questions + crisp answers

**Q: Why not just retrieve for every question?**
A: Wastes latency/tokens on questions the model already knows, and a single fixed search
under-serves multi-hop questions. Routing matches effort to complexity.

**Q: How does the router decide?**
A: One Groq call with a classification prompt → easy/medium/hard. Temperature 0 for
stability. Defensive parsing (extracts the label even from prose) and a fail-safe to
"medium" (retrieve) if it can't parse a label.

**Q: What if the router is wrong?**
A: Bounded failure. Wrong "easy" → a possibly-ungrounded answer (mitigated by conservative
prompt). Wrong "hard" → extra searches but a *correct* answer. We bias the fail-safe
toward retrieving. Long-term you'd log routes and measure accuracy (Phase 7 test set).

**Q: Why LangGraph over LangChain / a plain loop?**
A: The critique→search **cycle** with a conditional edge. LangChain chains are DAGs;
expressing a bounded loop cleanly (with shared state and a branch function) is exactly
what LangGraph adds. A hand-rolled `while` loop works but loses the declarative graph you
can visualize and unit-test.

**Q: How do you guarantee the loop terminates?**
A: `iterations` is incremented in the search node; the conditional edge only loops while
`iterations < 3`. It's a pure function of state, so termination is provable and testable
without any LLM.

**Q: Why 256-token chunks?**
A: The embedding model (`all-MiniLM-L6-v2`) truncates at 256 tokens. A 512-token chunk
would only be half-represented in its vector — silent truncation that degrades retrieval.
256 makes the embedding honest; I bumped top-k to 6 to keep enough context downstream.

**Q: Why FAISS `IndexFlatIP` and not IVF/HNSW?**
A: Corpus is a handful of papers → exact flat search is instant and has zero tuning.
IVF/HNSW are for millions of vectors; here they'd add approximation error for no benefit.
Normalized vectors make inner product == cosine similarity.

**Q: How are citations produced — can the model fake them?**
A: We number the evidence [1..n], instruct the model to cite inline `[n]`, then resolve
those markers back to the *actual* retrieved chunks (doc + page). The UI shows the real
sources, not model-invented ones. Claims without evidence are what the critique node
catches in Mode 3.

**Q: What's the failure mode if a paper isn't in the corpus?**
A: Generate says it lacks evidence rather than hallucinating; in Mode 3 the critique flags
unsupported claims, the loop retries up to 3×, then exits honestly at the cap.

**Q: How would you productionize this (AWS)?**
A: Out of scope for the demo, but the mapping is clean: FAISS → OpenSearch/pgvector or a
managed vector DB; Groq → Bedrock or a hosted endpoint; ingestion → a Lambda/batch job
writing to S3 + the index; Streamlit → a real frontend + an API. The **graph logic is
unchanged** — only the I/O boundaries move. That separation is deliberate.

**Q: Biggest risk / what would you improve next?**
A: Router accuracy is the leverage point — I'd build an eval set (started in Phase 7),
measure per-class precision/recall, and consider few-shot examples or a small fine-tuned
classifier if the LLM router drifts. Second: heading detection is heuristic; a layout-
aware parser would improve section-level citations.

---

## 12. 30-second whiteboard version

"Router classifies the question → three modes. Easy = answer directly, no search. Medium =
one FAISS search then generate with citations. Hard = decompose into sub-questions, search
each, generate, then a critique node checks every claim against the evidence; a conditional
edge loops back to search if claims are unsupported, capped at 3 iterations so it always
terminates. FAISS + local MiniLM embeddings for retrieval, Groq for the LLM steps, all
wired in LangGraph, fully local, no cloud."
```
```
