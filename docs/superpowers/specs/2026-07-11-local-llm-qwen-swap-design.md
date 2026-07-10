# Replace Groq with a local Qwen model via Ollama

## Context

This project has repeatedly hit Groq's daily token quota (100,000 tokens/day, free
tier) during development and live verification — most recently blocking the final
Mode 3 live-verification step of the general-document-support plan (Task 6), which
needed to wait ~72 minutes for a daily reset. The user wants this dependency removed
entirely: swap to a small model that runs locally on this MacBook Pro (18GB RAM,
arm64), no external API, no rate limit, no waiting.

Every LLM call in this project already goes through a single shared function,
`src.nodes.llm.chat(system, user, *, temperature=None, max_tokens=None) -> str`,
used identically by all four LLM-calling nodes (`router.py`, `direct_answer.py`,
`generate.py`, `decompose.py`, `critique.py`). This means the swap is scoped to one
file plus config/dependency bookkeeping — no changes needed to any node's logic.

Verified before designing: Ollama is already installed on this machine
(`/usr/local/bin/ollama`), with one model already pulled (`mistral:latest`, unrelated
to this project). No `qwen2.5:7b` model is pulled yet. The `ollama` Python package is
not yet installed in the project's venv.

## Decision: full replacement, Ollama's native Python package, Qwen 2.5 7B

- **Full replacement, not a fallback.** Per the user's explicit request, Groq is
  removed entirely from `llm.py`, `config.py`, and `requirements.txt` — no dual-path
  config flag, no kept-around fallback. If cloud LLM access is wanted again later,
  that is a separate future decision.
- **Model: `qwen2.5:7b`.** Fits comfortably in 18GB RAM (Q4 quantized, ~4.7GB) with
  headroom for the embedding model and FAISS index running alongside it, and is fast
  enough for interactive/live-demo use. This project's four call types (single-word
  classification, short sub-question generation, longer cited-answer generation,
  claim verification) don't need a larger model's extra reasoning depth — all four
  are well within a 7B instruction-tuned model's capability, especially at
  `temperature=0.0`.
- **Integration via Ollama's native Python package (`ollama`), not the OpenAI-
  compatible REST endpoint.** Simpler dependency, officially maintained, and its
  response shape maps cleanly onto `chat()`'s existing `-> str` contract with
  minimal translation. No benefit to the REST-compatibility layer here since this
  project isn't planning to swap local-serving tools again.
- **No retry/timeout scaffolding.** The current `llm.py` has Groq-specific
  `max_retries=5, timeout=30.0` added earlier specifically to absorb Groq's rate
  limits. A local model has no rate limit to retry through — a failure means the
  Ollama daemon isn't running or the model isn't pulled, and retrying won't fix
  either. Instead: raise a clear, actionable error (in the same spirit as today's
  `LLMConfigError` for a missing Groq key) telling the user to run `ollama serve`
  and/or `ollama pull qwen2.5:7b` if that's the failure. No artificial timeout —
  a local model's completion time is bounded by hardware speed, not an external
  API's availability, so an arbitrary timeout would only risk cutting off a slower
  but otherwise-successful local generation.

## Changes

### `src/nodes/llm.py`

- Remove `from groq import Groq` (moved inside `_client()` today; the whole function
  changes).
- `_client()` becomes a thin accessor that confirms Ollama is reachable (e.g. a
  lightweight `ollama.list()` call or equivalent) and returns the `ollama` module
  itself (or a minimal wrapper) rather than constructing an API-keyed client object
  — there is no key to construct with.
- `LLMConfigError` (name kept — it's a fitting label for "LLM isn't configured/
  reachable," not specific to Groq's API-key concept) is raised when Ollama isn't
  reachable, with a message telling the user to run `ollama serve` and
  `ollama pull qwen2.5:7b`.
- `chat(system, user, *, temperature=None, max_tokens=None) -> str` keeps its exact
  existing signature — every calling node is unaware this changed. Internally calls
  `ollama.chat(model=config.OLLAMA_MODEL, messages=[{"role": "system", ...}, ...],
  options={"temperature": ..., "num_predict": ...})` and returns
  `response["message"]["content"].strip()`.

### `src/config.py`

- Remove `GROQ_API_KEY`, `GROQ_MODEL`.
- Add `OLLAMA_MODEL = "qwen2.5:7b"`.
- `LLM_TEMPERATURE`, `LLM_MAX_TOKENS` are unchanged in meaning — `chat()` maps
  `max_tokens` onto Ollama's `options={"num_predict": ...}` parameter internally.
- Remove the `load_dotenv()`-driven `os.getenv("GROQ_API_KEY", "")` line — no key to
  load from `.env` for this purpose (the `.env` mechanism itself can stay for any
  future use, but this specific line goes).

### `requirements.txt`

- Remove `groq`.
- Add `ollama` (the Python package, not the CLI/daemon — that's a system-level
  install already present on this machine, not a pip dependency).

### `.env.example`

- Remove the `GROQ_API_KEY=your_groq_api_key_here` line and its accompanying
  "get a free key" comment.
- Replace with a short comment noting that this project now uses a local Ollama
  model and requires `ollama serve` running plus `ollama pull qwen2.5:7b` once,
  with no API key needed.

### No changes to any node file

`router.py`, `direct_answer.py`, `generate.py`, `decompose.py`, `critique.py` all
call `chat()` with the exact same signature — none of them need to change. This is
the direct payoff of the shared-helper design already in place.

## Setup step (action, not code)

Before this can be tested: `ollama pull qwen2.5:7b` (one-time download, ~4.7GB) and
confirm `ollama serve` is running (or that Ollama's background service is already
active, which is the default on macOS after installation).

## Testing approach

1. **Offline unit tests** (`tests/test_decompose.py`, `tests/test_critique.py`,
   `tests/test_graph.py`, `tests/test_ingest_headings.py`, `tests/test_ingest_title.py`,
   `tests/test_ingest_match_document.py`, `tests/test_search.py`) — none of these
   call `chat()` directly (they test parsing/branching logic against hand-built
   strings), so all must continue passing unchanged. This is the regression
   guarantee that the swap didn't break anything unrelated.
2. **Live verification against Qwen**, replacing every place this project
   previously live-verified against Groq:
   - Router: the same three spot-check questions used throughout this project's
     history (easy/medium/hard), confirming Qwen classifies them the same way Groq
     did — acknowledging Qwen's classification *judgment* may differ from Groq's on
     some questions (it's a different, smaller model), so this checks for
     *reasonable* classification, not byte-identical output to Groq.
   - Decompose, generate, critique: the same spot-checks used in Phases 3-4's
     BUILD_LOG entries, run against Qwen instead of Groq.
   - Full graph, all three modes, using `build_graph()` — this is also where the
     previously-paused Task 6 (general-document-support plan) gets completed: the
     Mode 3 end-to-end run that was blocked on Groq's quota, now run on Qwen
     instead. Per the user's decision, finishing Task 6 on Qwen is equally valid
     proof of that plan's correctness, since Task 6 verifies the system's behavior
     (heading detection, document matching, graph routing), not any specific LLM
     provider.
3. **Error-path check:** confirm `chat()` raises a clear error (not a cryptic
   traceback) if Ollama's daemon is stopped, by temporarily stopping it
   (`ollama stop` or killing the service) and attempting a call, then restarting and
   confirming recovery — mirroring the error-path verification already done for the
   Streamlit UI in Phase 6, adapted to the new failure mode.
4. **`app.py` requires no code change**, but its existing `st.error(...)` handling
   should be re-confirmed to show a clean message for this new error class, same as
   it did for Groq's errors.

## Out of scope

- Any change to `app.py`'s code (only re-verified, not modified, per Testing item 4).
- Any change to the retry/timeout design that was previously added for Groq — it is
  removed, not adapted, per the Decision section above.
- Comparing Qwen's answer quality/accuracy against Groq's in any rigorous way beyond
  "does it produce reasonable, correctly-shaped output" — a full quality comparison
  between the two models is not the goal here; the goal is removing the Groq
  dependency while keeping the system functionally correct.
- Any change to `src/ingest.py` or the FAISS/embedding pipeline — this swap is
  scoped entirely to the LLM-calling layer, not the retrieval layer.
