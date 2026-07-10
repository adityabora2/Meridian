# Phase 5 — Graph Assembly Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `src/graph.py`, wiring the six existing nodes
(`router`, `direct_answer`, `search`, `generate`, `decompose`, `critique`) into a
compiled LangGraph `StateGraph` implementing all three modes and the self-critique
retry loop, per the approved spec at
`docs/superpowers/specs/2026-07-10-phase5-graph-assembly-design.md`.

**Architecture:** One new file, `src/graph.py`. A router-conditional-edge branches to
Mode 1 (`direct_answer` → END), Mode 2 (`search` → `generate` → END), or Mode 3
(`decompose` → `search` → `generate` → `critique`, then a second conditional edge
that either ends cleanly, loops back through a new inline `prepare_retry` glue node
into `search`, or ends at the iteration cap). No existing node file changes.

**Tech Stack:** `langgraph==0.2.60` (already installed), confirmed API:
`from langgraph.graph import StateGraph, START, END`; `StateGraph(RAGState)`;
`.add_node(name: str, fn: Callable)`; `.add_conditional_edges(source: str, path_fn: Callable, path_map: dict)`;
`.add_edge(start: str, end: str)`; `.compile()` returns a `CompiledStateGraph` with
`.invoke(initial_state: dict) -> dict`.

## Global Constraints

- No changes to any file under `src/nodes/` or to `src/state.py`. Only `src/graph.py`
  is created.
- `RAGState` is used as-is with LangGraph's default overwrite merge semantics — no
  `Annotated`/reducer types on any field. Every node already returns its complete
  computed value for each field it touches (verified in Phases 3–4).
- The `prepare_retry` node lives inline in `src/graph.py`, not under `src/nodes/` — it
  makes no Groq call and has no independent responsibility beyond graph wiring:
  `{"sub_questions": state["unsupported_claims"]}`.
- Iteration cap: `config.MAX_ITERATIONS` (already `3`). No new counter — `search_node`
  (unchanged) already increments `state["iterations"]` every call; `route_question`
  (unchanged) already resets it to `0` at the start of every run.
- Route labels come from `config.ROUTE_EASY` / `config.ROUTE_MEDIUM` / `config.ROUTE_HARD`
  (values `"easy"` / `"medium"` / `"hard"`) — never hardcode the strings directly in
  `graph.py`; import and use the constants.
- Tests for the two conditional-edge branch functions are plain `assert`-based scripts
  run via `python -m` or direct invocation (no pytest), matching this repo's existing
  test style (`tests/test_decompose.py`, `tests/test_critique.py`).
- Live end-to-end graph dry-runs (Task 3) require a real `GROQ_API_KEY` in `.env`
  (already configured, confirmed working in Phase 3/4) and the real FAISS index
  (already built in Phase 2, `index/faiss.index` + `index/metadata.json` present).

---

### Task 1: Branch functions — `route_from_router` and `route_from_critique` (offline, no Groq)

**Files:**
- Create: `src/graph.py` (this task writes only the two branch functions + their
  required imports; the full graph assembly is Task 2)
- Test: `tests/test_graph.py`

**Interfaces:**
- Consumes: `src.state.RAGState`; `src.config.ROUTE_EASY`, `ROUTE_MEDIUM`, `ROUTE_HARD`,
  `MAX_ITERATIONS`.
- Produces: `route_from_router(state: RAGState) -> str` returning one of
  `"direct_answer"`, `"search"`, `"decompose"` (the three literal node names Task 2
  will register). `route_from_critique(state: RAGState) -> str` returning one of
  `"prepare_retry"`, `"end"` (Task 2 maps `"end"` to LangGraph's `END` sentinel via
  the `path_map` dict passed to `add_conditional_edges` — the function itself returns
  plain strings, never the `END` object, so it stays unit-testable without importing
  LangGraph's graph machinery).

This task is pure logic, no LangGraph graph construction yet — just the two functions
`add_conditional_edges` will call, unit-tested directly with hand-built state dicts.

- [ ] **Step 1: Write the failing test**

Create `tests/test_graph.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph import route_from_router, route_from_critique


def test_route_from_router_easy_goes_to_direct_answer():
    assert route_from_router({"route": "easy"}) == "direct_answer"


def test_route_from_router_medium_goes_to_search():
    assert route_from_router({"route": "medium"}) == "search"


def test_route_from_router_hard_goes_to_decompose():
    assert route_from_router({"route": "hard"}) == "decompose"


def test_route_from_critique_clean_ends():
    state = {"critique_clean": True, "iterations": 1}
    assert route_from_critique(state) == "end"


def test_route_from_critique_unsupported_below_cap_retries():
    state = {"critique_clean": False, "iterations": 2}
    assert route_from_critique(state) == "prepare_retry"


def test_route_from_critique_unsupported_at_cap_ends():
    state = {"critique_clean": False, "iterations": 3}
    assert route_from_critique(state) == "end"


def test_route_from_critique_boundary_two_below_three_cap():
    # iterations == MAX_ITERATIONS - 1 must still allow one more retry.
    state = {"critique_clean": False, "iterations": 2}
    assert route_from_critique(state) == "prepare_retry"


if __name__ == "__main__":
    test_route_from_router_easy_goes_to_direct_answer()
    test_route_from_router_medium_goes_to_search()
    test_route_from_router_hard_goes_to_decompose()
    test_route_from_critique_clean_ends()
    test_route_from_critique_unsupported_below_cap_retries()
    test_route_from_critique_unsupported_at_cap_ends()
    test_route_from_critique_boundary_two_below_three_cap()
    print("=== all graph branch-function tests PASSED ===")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/apple/Desktop/Projects/aws-rag && source venv/bin/activate && python tests/test_graph.py`
Expected: `ModuleNotFoundError: No module named 'src.graph'`

- [ ] **Step 3: Write `src/graph.py` (branch functions only)**

```python
from __future__ import annotations

try:
    from src import config
    from src.state import RAGState
except ImportError:
    import config  # type: ignore
    from state import RAGState  # type: ignore


def route_from_router(state: RAGState) -> str:
    route = state["route"]
    if route == config.ROUTE_EASY:
        return "direct_answer"
    if route == config.ROUTE_MEDIUM:
        return "search"
    return "decompose"


def route_from_critique(state: RAGState) -> str:
    if state.get("critique_clean", True):
        return "end"
    if state.get("iterations", 0) < config.MAX_ITERATIONS:
        return "prepare_retry"
    return "end"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/apple/Desktop/Projects/aws-rag && source venv/bin/activate && python tests/test_graph.py`
Expected: `=== all graph branch-function tests PASSED ===`

- [ ] **Step 5: Commit**

```bash
git add src/graph.py tests/test_graph.py
git commit -m "Add graph branch functions for router and critique routing"
```

---

### Task 2: Full graph assembly — `build_graph()` and `prepare_retry`

**Files:**
- Modify: `src/graph.py` (add `prepare_retry`, `build_graph()`, node registration,
  edge wiring — append to the file Task 1 created, do not remove the branch
  functions)

**Interfaces:**
- Consumes: `route_from_router`, `route_from_critique` (Task 1, same file);
  `src.nodes.router.route_question`, `src.nodes.direct_answer.direct_answer`,
  `src.nodes.search.search_node`, `src.nodes.generate.generate`,
  `src.nodes.decompose.decompose`, `src.nodes.critique.critique` — all
  `Callable[[RAGState], RAGState]` (dict-in, partial-dict-out), already built and
  reviewed in Phases 3–4. `langgraph.graph.StateGraph`, `START`, `END`.
- Produces: `build_graph() -> CompiledStateGraph` — a module-level function
  (not a global variable) so Task 3's live tests and any future caller (Phase 6's
  Streamlit app) construct their own instance rather than sharing mutable state.
  The returned object supports `.invoke(initial_state: dict) -> dict` (LangGraph's
  standard compiled-graph interface).

- [ ] **Step 1: Add the node functions and `build_graph()` to `src/graph.py`**

Append to `src/graph.py` (after the two functions from Task 1), replacing the
existing import block at the top of the file with this expanded version:

```python
from __future__ import annotations

try:
    from src import config
    from src.nodes.critique import critique
    from src.nodes.decompose import decompose
    from src.nodes.direct_answer import direct_answer
    from src.nodes.generate import generate
    from src.nodes.router import route_question
    from src.nodes.search import search_node
    from src.state import RAGState
except ImportError:
    import config  # type: ignore
    from nodes.critique import critique  # type: ignore
    from nodes.decompose import decompose  # type: ignore
    from nodes.direct_answer import direct_answer  # type: ignore
    from nodes.generate import generate  # type: ignore
    from nodes.router import route_question  # type: ignore
    from nodes.search import search_node  # type: ignore
    from state import RAGState  # type: ignore


def route_from_router(state: RAGState) -> str:
    route = state["route"]
    if route == config.ROUTE_EASY:
        return "direct_answer"
    if route == config.ROUTE_MEDIUM:
        return "search"
    return "decompose"


def route_from_critique(state: RAGState) -> str:
    if state.get("critique_clean", True):
        return "end"
    if state.get("iterations", 0) < config.MAX_ITERATIONS:
        return "prepare_retry"
    return "end"


def prepare_retry(state: RAGState) -> RAGState:
    return {"sub_questions": state["unsupported_claims"]}


def build_graph():
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(RAGState)

    graph.add_node("router", route_question)
    graph.add_node("direct_answer", direct_answer)
    graph.add_node("decompose", decompose)
    graph.add_node("search", search_node)
    graph.add_node("generate", generate)
    graph.add_node("critique", critique)
    graph.add_node("prepare_retry", prepare_retry)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_from_router,
        {"direct_answer": "direct_answer", "search": "search", "decompose": "decompose"},
    )
    graph.add_edge("direct_answer", END)
    graph.add_edge("decompose", "search")
    graph.add_edge("search", "generate")
    graph.add_conditional_edges(
        "generate",
        lambda state: "critique" if state.get("sub_questions") else "end_mode2",
        {"critique": "critique", "end_mode2": END},
    )
    graph.add_conditional_edges(
        "critique",
        route_from_critique,
        {"prepare_retry": "prepare_retry", "end": END},
    )
    graph.add_edge("prepare_retry", "search")

    return graph.compile()
```

**Design note on the `generate` → `critique`-or-`END` branch:** Mode 2 and Mode 3 both
route through `search` → `generate`, but only Mode 3 continues to `critique`. The
distinguishing signal already in state is `sub_questions`: Mode 2 never sets it
(single-hop, `search_node` falls back to `[state["question"]]` internally per its
existing code), Mode 3 always does (via `decompose`, and later `prepare_retry`). This
inline lambda is simple enough not to warrant a named function, but if you find it
unclear while implementing, promote it to a named `route_from_generate` function
following the same pattern as the other two branch functions — either is acceptable,
prefer whichever reads more clearly once the full file is in front of you.

- [ ] **Step 2: Verify the graph compiles**

Run:

```bash
cd /Users/apple/Desktop/Projects/aws-rag && source venv/bin/activate && python -c "
from src.graph import build_graph
g = build_graph()
print('Graph compiled OK:', type(g).__name__)
"
```

Expected: `Graph compiled OK: CompiledStateGraph` (no exceptions — a compile-time
error here usually means a node name referenced in `path_map` doesn't match an
`add_node` name, or an edge points to an unregistered node).

- [ ] **Step 3: Re-run Task 1's tests to confirm no regression**

Run: `cd /Users/apple/Desktop/Projects/aws-rag && source venv/bin/activate && python tests/test_graph.py`
Expected: `=== all graph branch-function tests PASSED ===` (unchanged from Task 1 —
this task only added code, the two original functions are untouched).

- [ ] **Step 4: Commit**

```bash
git add src/graph.py
git commit -m "Wire nodes into a compiled LangGraph StateGraph"
```

---

### Task 3: Live end-to-end dry-runs for all three modes

**Files:**
- None created/modified — this task runs `build_graph()` against the real Groq API
  and the real FAISS index (both already configured/built in Phases 2–4), then
  records results in `BUILD_LOG.md` and flips Phase 5's checkbox in `PLAN.md`.

**Interfaces:**
- Consumes: `build_graph()` from Task 2. `GROQ_API_KEY` from `.env` (already
  confirmed working). `index/faiss.index` + `index/metadata.json` (already built from
  the two arXiv papers in Phase 2).

This is a manual verification task (live LLM calls), not a TDD unit-test task — same
pattern as Phase 3's router spot-check (BUILD_LOG entry 8) and Phase 4's node
spot-checks (BUILD_LOG entry 9).

- [ ] **Step 1: Dry-run Mode 1 (easy, zero retrieval)**

Run:

```bash
cd /Users/apple/Desktop/Projects/aws-rag && source venv/bin/activate && python -c "
from src.graph import build_graph

g = build_graph()
result = g.invoke({'question': 'What is a vector database?', 'trace': []})
print('Route:', result.get('route'))
print('Mode:', result.get('mode_label'))
print('Answer:', result.get('answer'))
print('Citations:', result.get('citations'))
print('Trace:', result.get('trace'))
"
```

Expected: `Route: easy`, `Mode: Mode 1 · No Retrieval`, a plausible direct answer,
`Citations: []`, and the trace shows only `router` → `direct_answer` (no `search`/
`generate` entries).

- [ ] **Step 2: Dry-run Mode 2 (medium, single-hop)**

Run:

```bash
cd /Users/apple/Desktop/Projects/aws-rag && source venv/bin/activate && python -c "
from src.graph import build_graph

g = build_graph()
result = g.invoke({'question': 'What optimizer does the Transformer use for training?', 'trace': []})
print('Route:', result.get('route'))
print('Mode:', result.get('mode_label'))
print('Answer:', result.get('answer'))
print('Citations:', result.get('citations'))
print('Iterations:', result.get('iterations'))
print('Trace:', result.get('trace'))
"
```

Expected: `Route: medium`, `Mode: Mode 2 · Single-Hop Retrieval`, an answer citing the
Adam optimizer with at least one citation pointing at `attention_is_all_you_need.pdf`,
`Iterations: 1`, and the trace shows `router` → `search` → `generate` (no
`decompose`/`critique`).

- [ ] **Step 3: Dry-run Mode 3 (hard, multi-hop + critique loop)**

Run:

```bash
cd /Users/apple/Desktop/Projects/aws-rag && source venv/bin/activate && python -c "
from src.graph import build_graph

g = build_graph()
result = g.invoke({
    'question': 'Compare how BERT and the Transformer paper each handle positional information, and explain any tradeoffs.',
    'trace': [],
})
print('Route:', result.get('route'))
print('Mode:', result.get('mode_label'))
print('Answer:', result.get('answer'))
print('Citations:', result.get('citations'))
print('Iterations:', result.get('iterations'))
print('Critique clean:', result.get('critique_clean'))
print('Unsupported claims:', result.get('unsupported_claims'))
print('Trace:')
for t in result.get('trace', []):
    print(' -', t)
"
```

Expected: `Route: hard`, `Mode: Mode 3 · Multi-Hop + Self-Critique`, an answer
comparing both papers' positional-encoding approaches with citations from both PDFs,
`Iterations` between 1 and 3 inclusive (**never more than 3** — this is the property
that must hold), and the trace shows `router` → `decompose` → `search` → `generate` →
`critique`, optionally repeated (`prepare_retry` → `search` → `generate` → `critique`)
up to the cap. If `Iterations == 3` and `Critique clean == False`, that is the
expected cap-hit termination, not a bug — confirm the trace shows exactly 3
`search` entries and the graph still returned a result instead of hanging or raising.

- [ ] **Step 4: Update `PLAN.md`**

Read `PLAN.md`, find `## Phase 5 — Assemble the graph (\`src/graph.py\`)  ☐  *(the core deliverable)*`
and its body. Change the checkbox to `☑`, and add the three dry-run results
underneath, following the same style as Phases 2–4's updates (status line + concrete
observed results, e.g. the exact iteration count seen in Step 3, whether it hit clean
or hit the cap).

- [ ] **Step 5: Add a `BUILD_LOG.md` entry**

Append a new entry (next sequential number after entry 9) documenting: what was built
(`src/graph.py` — the branch functions, `prepare_retry`, `build_graph()`), why (Phase
5 per PLAN.md, the core deliverable — unblocks Phase 6's Streamlit UI), the offline
branch-function test results (Task 1, all passed including the iteration-cap
boundary case), and the live end-to-end results for all three modes (Task 3 — exact
observations: which mode fired, citation counts, and critically, the exact iteration
count and clean/cap-hit outcome for the Mode 3 run).

- [ ] **Step 6: Commit**

```bash
git add PLAN.md BUILD_LOG.md
git commit -m "Complete Phase 5: graph assembly verified live across all 3 modes"
```

---

## Self-Review Notes

- **Spec coverage:** router conditional branch (Task 1 `route_from_router` + Task 2
  wiring) — done. Mode 2 path (`search` → `generate` → END) — Task 2. Mode 3 path
  (`decompose` → `search` → `generate` → `critique`) — Task 2. Critique conditional
  edge with iteration cap (Task 1 `route_from_critique`, boundary-tested at
  `iterations == 2` and `== 3`) — done. `prepare_retry` glue node — Task 2. Plain
  overwrite state semantics — no `Annotated` reducers appear anywhere in the plan's
  code. Live dry-run of all 3 modes — Task 3. All spec sections covered.
- **Placeholder scan:** no TBD/TODO; every step has complete runnable code; the one
  spot where two valid implementations exist (inline lambda vs. named function for
  the generate→critique branch) is explicitly flagged as an either-is-fine judgment
  call, not left ambiguous as a gap.
- **Type consistency:** `route_from_router(state: RAGState) -> str`,
  `route_from_critique(state: RAGState) -> str`, `prepare_retry(state: RAGState) -> RAGState`,
  and `build_graph() -> CompiledStateGraph` are used identically between Task 1, Task
  2's full file, and Task 3's invocations (`g.invoke(...)`). Node name strings
  (`"direct_answer"`, `"search"`, `"decompose"`, `"critique"`, `"prepare_retry"`) match
  exactly between `add_node` calls, `path_map` dicts, and the branch functions'
  return values in Task 2 Step 1.
