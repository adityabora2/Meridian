# Phase 5 — Assemble the graph (`src/graph.py`)

## Context

Phases 0–4 built every node in isolation: `router.py`, `direct_answer.py`,
`search.py`, `generate.py` (Phase 3, Mode 1/2), and `decompose.py`, `critique.py`
(Phase 4, Mode 3 building blocks). Per PLAN.md, Phase 5 is "the core deliverable" —
wiring these six existing nodes into an actual LangGraph `StateGraph` with the
router's conditional branch and the self-critique retry loop with its 3-iteration
cap. No node's internal logic changes in this phase; this is pure graph assembly.

## Graph shape

**Entry:** `router`

**Conditional edge after `router`**, branching on `state["route"]`
(`config.ROUTE_EASY` / `ROUTE_MEDIUM` / `ROUTE_HARD`):
- `easy` → `direct_answer` → END (Mode 1, zero retrieval)
- `medium` → `search` → `generate` → END (Mode 2, single-hop)
- `hard` → `decompose` → `search` → `generate` → `critique` (Mode 3, multi-hop)

**Conditional edge after `critique`**, branching on `state["critique_clean"]` and
`state["iterations"]`:
- `critique_clean is True` → END
- `critique_clean is False` and `iterations < config.MAX_ITERATIONS` → `prepare_retry`
  → `search` (loops back into the search→generate→critique cycle)
- otherwise (cap hit) → END

## New glue node: `prepare_retry`

A minimal node defined **inline in `src/graph.py`**, not under `src/nodes/`, because
it makes no Groq call and has no independent responsibility beyond graph wiring:

```python
def prepare_retry(state: RAGState) -> RAGState:
    return {"sub_questions": state["unsupported_claims"]}
```

This is the mechanism that turns critique's `unsupported_claims` (already phrased as
search-friendly follow-up questions, per the Phase 4 design) into the next
iteration's `sub_questions` for `search_node` to consume. `search_node` itself
requires zero changes — it already just reads whatever is in `state["sub_questions"]`.

## Iteration cap enforcement

No new counter is introduced. `search_node` (Phase 3, unchanged) already increments
`state["iterations"]` on every call, and `route_question` (Phase 3, unchanged)
already resets `iterations=0` at the start of every run. The conditional edge
function after `critique` reads `state["iterations"]` against
`config.MAX_ITERATIONS` (already `3`) to decide whether to loop or terminate. This
guarantees the loop runs at most 3 search→generate→critique cycles per question,
regardless of how many times critique reports `unsupported`.

## State merge semantics

`RAGState` (already defined in `src/state.py`, unchanged) is used as a plain
`TypedDict` with LangGraph's default overwrite behavior — **no `Annotated` reducers**
on any field, including list fields like `trace` and `retrieved`. Every existing node
already performs its own read-modify-write and returns the complete new value (e.g.
`trace = list(state.get("trace", [])); trace.append(...); return {"trace": trace}`;
`search_node` returns the full pooled+deduped `retrieved` list, not just new hits).
Adding an `operator.add`-style reducer would double-accumulate on top of what nodes
already merge internally. Plain overwrite is correct and requires no changes to any
node.

## No changes to existing node files

`router.py`, `direct_answer.py`, `search.py`, `generate.py`, `decompose.py`,
`critique.py` are used exactly as built in Phases 3–4. Only `src/graph.py` is new.

## Testing approach

Per PLAN.md's Phase 5 exit check — compile the graph, then dry-run all three modes:

1. **Mode 1 (easy):** a general-knowledge question. Confirm the compiled graph routes
   to `direct_answer` only, `citations == []`, no `search`/`generate` nodes execute.
2. **Mode 2 (medium):** a question needing one retrieval. Confirm `search` → `generate`
   executes once, `answer` and `citations` are populated, graph ends without touching
   `decompose`/`critique`.
3. **Mode 3 (hard):** a genuinely multi-hop question. Confirm the graph either exits
   `clean` before the cap, or iterates until `iterations == config.MAX_ITERATIONS`
   and then terminates — never more than `MAX_ITERATIONS` search→generate→critique
   cycles. This is the property PLAN.md calls "interview-ready": the loop must
   provably terminate.

All three are live-Groq tests (every node but `prepare_retry` calls `chat()` or
`faiss_search`), run the same way Phase 3/4's live spot-checks were run — via a
`python -m src.graph` or ad-hoc script invocation, not unit tests with mocked LLM
responses. No new offline/mocked test suite is needed for `graph.py` itself since it
contains no parsing logic of its own to unit-test in isolation — its only "logic" is
the two conditional-edge branch functions, which are pure and can be unit-tested
directly with hand-built state dicts (does `iterations < MAX_ITERATIONS` route
correctly at the boundary, e.g. `iterations == 2` loops, `iterations == 3` doesn't).

## Out of scope for Phase 5

- Any change to `src/nodes/*.py`.
- The Streamlit UI (`app.py`, Phase 6).
- The 30-question test set (Phase 7).
