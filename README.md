# Meridian — Adaptive RAG

A document Q&A system that decides *how* to retrieve **before** it retrieves —
instead of running every query through the same fixed search-then-generate pipeline.

Runs fully locally: a local LLM (via Ollama), local embeddings, and a disk-persisted
FAISS index. No cloud services, no API keys.

## The idea

Most RAG systems follow a single path regardless of query complexity:

```
Question → Search → Retrieve → Generate
```

- Simple questions get over-retrieved: noisy context, worse answers.
- Complex multi-hop questions get under-retrieved: one search isn't enough.

Meridian classifies each query's complexity *first*, then routes it down one of three
paths:

- **Mode 1 — Direct Answer.** General-knowledge questions the model can answer without
  looking at any document. No retrieval.
- **Mode 2 — Single-Hop Retrieval.** One vector search, then grounded generation with
  inline citations (document + page).
- **Mode 3 — Multi-Hop + Self-Critique.** Decomposes the question into sub-questions,
  retrieves evidence for each, generates an answer, then verifies every claim against
  the retrieved evidence — looping back to search if a claim isn't supported, capped at
  3 iterations so it always terminates.

The routing decision is shown in the UI, so it's clear which mode handled each query.

## Stack

| Layer | Tool |
|---|---|
| Orchestration | LangGraph |
| LLM | `qwen2.5:7b` served locally by [Ollama](https://ollama.com) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`), local |
| Vector store | FAISS (`IndexFlatIP`, disk-persisted) |
| PDF ingestion | PyMuPDF (256-token chunks, 50-token overlap) |
| Frontend | Streamlit |

## Documents

Meridian works with **general PDF documents**, not just research papers. Heading
detection is font-size-based (a heading is text visually larger than the body, not text
matching academic section names), and title extraction handles real-world quirks
(empty or junk embedded metadata, rotated preprint watermarks, small-caps titles). It's
been verified end-to-end on both academic papers and a non-academic document (the US
Constitution — legal structure like "Article V" / "Amendment XII" is detected
correctly).

## Setup

Requires Python 3.11+ and [Ollama](https://ollama.com) installed.

```bash
# 1. Install Ollama (see https://ollama.com), then pull the model:
ollama pull qwen2.5:7b

# 2. Create the environment and install dependencies:
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Add your PDFs:
#    Drop any PDF documents into data/documents/

# 4. Build the index (parses, chunks, embeds, persists to index/):
python -m src.ingest

# 5. Launch the app:
streamlit run app.py
```

No API key or `.env` configuration is needed — the LLM runs locally. Ollama must be
running (on macOS it runs as a background service after install; otherwise
`ollama serve`).

## Usage

Ask any question in the Streamlit UI. The router decides the mode automatically. For
questions about your documents, answers come with citations (document name + page); the
collapsible "How this answer was produced" panel shows the routed mode, iteration count,
and full execution trace.

## Testing

Offline unit tests (no LLM required — they cover the pure parsing/branching logic).
`test_questions.py` is the live harness (below), so it's excluded here:

```bash
source venv/bin/activate
for t in tests/test_*.py; do
  [ "$t" = "tests/test_questions.py" ] && continue
  python "$t"
done
```

Or, if you have `pytest` installed (a dev-only convenience, not a project dependency):
`pytest tests/ --ignore=tests/test_questions.py`.

Live routing-accuracy harness (requires Ollama running and an index built):

```bash
python -m tests.test_questions            # 30 labelled questions, routing accuracy
python -m tests.test_questions --mode3    # also runs hard questions through the full graph
```

The harness reports per-class and overall routing accuracy. On `qwen2.5:7b` the current
result is **28/30 (93.3%)** — easy 10/10, hard 10/10, medium 8/10 (the two medium misses
route to *hard*, which fails safe: more retrieval, not less).

## Project structure

```
src/
  config.py      model, chunking, top-k, iteration-cap settings
  ingest.py      PDF parse, heading/title extraction, chunk, embed, FAISS index, search
  state.py       shared RAGState (the graph's state contract)
  graph.py       LangGraph assembly: router branch, Mode 1/2/3 paths, critique retry loop
  nodes/
    llm.py           shared chat() helper (Ollama) — the single LLM entry point
    router.py        classify easy / medium / hard
    direct_answer.py Mode 1
    search.py        FAISS retrieval with per-document scoping
    generate.py      grounded generation with [n] citations
    decompose.py     Mode 3: split a hard question into sub-questions
    critique.py      Mode 3: verify claims against evidence
app.py           Streamlit UI
tests/           offline unit tests + the live routing harness (test_questions.py)
data/documents/  your PDFs (gitignored)
index/           persisted FAISS index + metadata (gitignored, rebuilt by ingest.py)
```

## Known limitations

- **Citation formatting on Mode 3 (local model).** `qwen2.5:7b` is less reliable than a
  larger model at emitting parseable `[n]` citation markers on long, multi-part Mode 3
  answers — the evidence is retrieved correctly, but citations sometimes fail to resolve.
  Mode 1/2 citations are reliable. A larger local model (e.g. a 14B) improves this if
  needed; the `--mode3` test flag tracks it as a measurable metric.
- **Cross-document scoping is conservative.** When a query clearly names one document,
  retrieval is scoped to it; otherwise it searches the whole corpus. Ambiguous queries
  fall back to whole-corpus search rather than risk scoping to the wrong document.
- **Scope.** No AWS, no auth, no multi-user — this is a single-user local system.
