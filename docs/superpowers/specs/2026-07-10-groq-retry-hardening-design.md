# Groq retry/backoff hardening (`src/nodes/llm.py`)

## Context

A production-readiness audit (run after expanding the corpus from 2 to 10 papers,
1490 chunks) identified that `src/nodes/llm.py`'s `chat()` function has no retry,
backoff, or timeout handling of its own — a bare `client.chat.completions.create(...)`
call. Mode 3 (multi-hop + self-critique) can make up to 8 Groq calls per question
(1 router + 1 decompose + up to 3 × (generate + critique) at the iteration cap),
making it the most rate-limit-exposed path in the system. The user reported hitting
a Groq rate limit earlier in this session, which is almost certainly this exact
failure mode.

Investigation during brainstorming found the Groq Python SDK (`groq==0.13.1`, an
OpenAI-SDK-derived client) already retries transient errors (429, 5xx, connection
errors) automatically with its own internal backoff — the constructor's
`max_retries` parameter defaults to `2`. So the SDK is not silently making zero
retry attempts; the gap is narrower than "no retry logic exists": it's (a) the
default retry budget is thin for a live demo, and (b) there is no `timeout` set, so
a hung request could stall indefinitely while the SDK's own retry loop is still in
flight.

## Decision: widen the existing SDK retry budget, do not build a custom retry loop

Per discussion with the user: after the SDK's retries are exhausted (a genuine,
sustained failure — bad key, fully exhausted quota, Groq outage), the system should
**fail loudly and clearly**, not mask the failure with a canned/degraded answer. A
demo silently returning a fake "answer unavailable" response is worse than a demo
that clearly shows an error — the existing `app.py` error path (verified live in
Phase 6: `st.error(f"Something went wrong: {exc}")` shows a clean message, not a raw
traceback) already does the right thing here and needs no change.

This means the fix is narrowly scoped to **giving the SDK's own retry mechanism more
room to recover from a transient/moderate rate limit** before giving up, not
building custom retry/backoff/circuit-breaker logic on top of it.

## Changes

### `src/nodes/llm.py`

In `_client()` (currently constructs `Groq(api_key=key)` with all defaults), pass:
- `max_retries=5` (up from the SDK default of `2`) — gives more headroom to recover
  from a short rate-limit window (Groq's free-tier limits are typically per-minute,
  so a handful of retries with the SDK's exponential backoff usually clears within
  one such window).
- `timeout=30.0` (seconds, explicit) — bounds how long a single request attempt can
  hang before the SDK's own retry/failure logic takes over, so a network stall
  doesn't block indefinitely underneath the retry budget.

No other logic in `llm.py` changes. `chat()`'s signature, its `temperature`/
`max_tokens` handling, and the `LLMConfigError` raised for a missing/placeholder key
are all unchanged.

### `app.py`

Update the spinner message only (`st.spinner("Running the adaptive RAG pipeline...")`
at the call site) to set expectations that a run may take longer than usual if a
transient rate limit is being retried underneath, e.g.:
`"Running the adaptive RAG pipeline... (this may take longer if the API is briefly rate-limited)"`.
No change to the `try/except`/`st.error` error-handling structure itself — it already
handles the fail-loudly case correctly.

## What this does NOT do

- Does not add a custom retry loop, exponential backoff implementation, or circuit
  breaker in application code — the SDK's built-in mechanism (now with a wider
  budget) is the single source of retry behavior.
- Does not add a fallback LLM, cached/canned response, or any form of degraded
  answer when Groq is genuinely unavailable after retries are exhausted.
- Does not change cross-paper retrieval contamination, heading-detection quality, or
  the router/decompose prompt's topic mismatch — those are separate audit findings,
  out of scope for this fix.
- Does not touch any node file other than `llm.py`, since every node already goes
  through the shared `chat()` helper — this is the single point of leverage.

## Testing approach

Deliberately exhausting a real Groq rate limit to test the retry path live is
impractical and wasteful (would require sustained hammering of the API). Instead:
1. **Configuration check:** confirm the `Groq` client is constructed with the new
   `max_retries=5` and `timeout=30.0` values (inspectable from the constructed
   client object or by confirming the exact call in the diff).
2. **Reuse the existing error-path test from Phase 6:** temporarily invalidate
   `GROQ_API_KEY`, confirm `chat()` still raises and `app.py`'s `st.error(...)`
   still shows a clean message (not a raw traceback) — this exercises the
   "exhausted/failed after retries" path end-to-end, since an invalid key fails
   immediately without retrying (a `401`, not a `429`), which is a reasonable proxy
   for "the failure path still works," even though it does not exercise the retry
   count itself.
3. A live, successful Groq call (any existing preset question) after the change
   confirms the happy path is unaffected by the wider retry/timeout settings.

## Out of scope (separate audit findings, not part of this fix)

- Cross-paper retrieval contamination (Mode 3 pooling searches all 10 papers with no
  document-level filtering).
- Heading-detection quality (86% of chunks mislabeled "Abstract"/"References"/
  "Acknowledgments").
- Router/decompose prompt referencing papers not in the corpus
  ("Adaptive-RAG, Self-RAG, Chain-of-Verification").
- Phase 7's 30-question test set and the router/search/generate test-coverage gap.
