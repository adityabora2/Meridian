# Phase 4 — Mode 3 building blocks: `decompose.py` + `critique.py`

## Context

Phases 0–3 built the scaffold, the ingestion pipeline, and the nodes for Mode 1
(no-retrieval) and Mode 2 (single-hop retrieval): `router.py`, `direct_answer.py`,
`search.py`, `generate.py`. Per PLAN.md, Phase 4 builds the two remaining nodes needed
for **Mode 3** (multi-hop + self-critique): `decompose.py` and `critique.py`. These
don't get wired into a graph yet — that's Phase 5. Phase 4 just needs both nodes working
correctly in isolation, producing the state fields the Phase-5 conditional edge will
branch on.

## `src/nodes/decompose.py`

Splits a hard question into 2-3 focused, independently-searchable sub-questions so
`search_node` (already built, `src/nodes/search.py`) can run one search per sub-question
via `state["sub_questions"]`.

- **Prompt:** system prompt instructs the LLM to produce 2-3 sub-questions, one per
  line, no numbering, no explanation — same terse-output style as `router.py`'s prompt.
- **Parsing:** split response on newlines, strip blank lines, cap at 3 even if the LLM
  over-produces. If parsing yields zero usable sub-questions, fall back to
  `[state["question"]]` (the original question as the sole "sub-question") — same
  fail-safe philosophy as the router's fallback to `medium`.
- **Output:** `{"sub_questions": [...], "trace": [...]}` (trace entry: e.g.
  `"decompose → 3 sub-question(s)"`).
- **Signature:** `decompose(state: RAGState) -> RAGState`, matching `route_question`'s
  and `search_node`'s calling convention (plain dict in, partial dict out).

## `src/nodes/critique.py`

Checks every claim in the draft answer against the retrieved evidence; produces the
signal the Phase-5 conditional edge reads to decide `clean → END` vs.
`unsupported + iterations < 3 → back to search`.

- **Inputs:** original question, draft answer (`state["answer"]`), and the same
  numbered evidence format `generate.py` already builds (reuse `_format_evidence` —
  move it to a shared location or duplicate the small helper, decide at implementation
  time based on which is cleaner).
- **Prompt:** given question + answer + numbered evidence, identify any claim in the
  answer not actually supported by the evidence. Response format uses sentinel lines
  (matches `router.py`'s plain-text house style, not JSON):
  ```
  VERDICT: clean
  ```
  or
  ```
  VERDICT: unsupported
  CLAIMS:
  - <unsupported claim, restated as a short question/topic>
  - <unsupported claim 2>
  ```
  Claims are phrased as short questions/topics (not verbatim quotes) because they
  become the next iteration's search queries (see Retry Wiring below) — they need to
  read as reasonable FAISS queries.
- **Parsing:** regex/line-based, defensive. If the `VERDICT` line is missing or
  unparseable, fail-safe to **`clean`** (stops the loop) rather than `unsupported` —
  a parse failure shouldn't force the loop to spin to the iteration cap; cheaper to
  under-critique once than to loop needlessly.
- **Output:** `{"critique_clean": bool, "unsupported_claims": [...], "trace": [...]}`
  (trace entry: e.g. `"critique → unsupported (2 claim(s))"`).
- **Signature:** `critique(state: RAGState) -> RAGState`.

## Retry wiring (context for Phase 5, shapes critique's output now)

Phase 5 will feed `unsupported_claims` back in as the next iteration's
`state["sub_questions"]`, so `search_node` searches specifically for the missing
evidence rather than blindly re-running the original sub-questions. This is why
critique's claims must be phrased as search-friendly questions/topics, not raw quoted
sentences from the answer.

## Testing approach

Same pattern as Phase 3: unit-test parsing logic offline with hand-built LLM response
strings (no Groq call) — decompose's newline-splitting + fallback, critique's
sentinel-line parsing + fallback-to-clean. Then one live Groq spot-check per node once
implemented (same shape as the Phase 3 router spot-check already logged in
BUILD_LOG.md entry 8): run `decompose()` on a real hard question and confirm 2-3
sensible sub-questions come back; run `critique()` on a hand-crafted answer containing
one deliberately unsupported claim and confirm it's flagged, and on a fully-grounded
answer and confirm it passes clean.

## Out of scope for Phase 4

- Wiring these nodes into `src/graph.py` (Phase 5).
- The iteration-cap enforcement logic itself (Phase 5, though `search_node` already
  increments `state["iterations"]`).
- Any change to `router.py`, `search.py`, or `generate.py`.
