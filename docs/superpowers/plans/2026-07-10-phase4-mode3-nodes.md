# Phase 4 — Mode 3 Building Blocks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `src/nodes/decompose.py` and `src/nodes/critique.py`, the two remaining
nodes needed for Mode 3 (multi-hop + self-critique), per the approved spec at
`docs/superpowers/specs/2026-07-10-phase4-mode3-nodes-design.md`.

**Architecture:** Both nodes follow the exact calling convention already established by
`src/nodes/router.py` and `src/nodes/generate.py`: a plain function
`node(state: RAGState) -> RAGState` that builds a system prompt, calls the shared
`chat()` helper from `src/nodes/llm.py`, defensively parses a plain-text (non-JSON)
response, and returns a partial state dict including an appended `trace` entry. No new
shared infrastructure is needed — `llm.chat()`, `RAGState`, and evidence formatting
already exist.

**Tech Stack:** Python 3.13, Groq (`llama-3.3-70b-versatile` via `src/nodes/llm.chat()`),
no new dependencies, no test framework (plain `assert`-based scripts run via
`python -m`, matching `src/ingest.py`'s `--self-test` pattern — this repo has no pytest
in `requirements.txt` and no committed test files yet).

## Global Constraints

- No new dependencies. Use only what's already in `requirements.txt`.
- Sentinel-line plain-text LLM response format (not JSON) — matches `router.py`'s house
  style, per the approved spec.
- Every parse failure must fail safe, never raise: decompose falls back to
  `[state["question"]]`; critique falls back to `critique_clean=True` (stops the loop
  rather than risking it spin to the cap on a parsing bug).
- `temperature=0.0` for both calls (classification/verification tasks) — this is already
  `config.LLM_TEMPERATURE`'s default in `llm.chat()`, so don't override it.
- Both nodes append to `state.get("trace", [])` and return it, matching every other node.
- Both files must support being run standalone via `if __name__ == "__main__":` with a
  `try/except ImportError` dual-import block, matching every existing file under
  `src/nodes/` (see `src/nodes/router.py:5-10` for the exact pattern to copy).

---

### Task 1: `decompose.py` — parsing + fallback logic (offline, no Groq)

**Files:**
- Create: `src/nodes/decompose.py`
- Test: `tests/test_decompose.py`

**Interfaces:**
- Consumes: `src.state.RAGState` (TypedDict, `total=False`, fields include `question: str`,
  `sub_questions: list[str]`, `trace: list[str]`).
- Produces: `decompose(state: RAGState) -> RAGState` returning
  `{"sub_questions": list[str], "trace": list[str]}`. Later tasks (Phase 5, out of
  scope here) will read `state["sub_questions"]` — same field `src/nodes/search.py:23`
  already reads.

This task builds the pure parsing logic (`_parse_sub_questions`) and unit-tests it with
hand-built strings, no live Groq call. The `decompose()` function itself (which calls
`chat()`) is written here too, since it's a thin wrapper, but only spot-checked live in
Task 3.

- [ ] **Step 1: Write the failing test for parsing**

Create `tests/test_decompose.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nodes.decompose import _parse_sub_questions


def test_parses_multiple_lines():
    raw = "How does Adaptive-RAG route questions?\nHow does Self-RAG use reflection tokens?"
    result = _parse_sub_questions(raw, fallback="orig question")
    assert result == [
        "How does Adaptive-RAG route questions?",
        "How does Self-RAG use reflection tokens?",
    ]


def test_strips_blank_lines():
    raw = "\n  First question?  \n\n\nSecond question?\n\n"
    result = _parse_sub_questions(raw, fallback="orig question")
    assert result == ["First question?", "Second question?"]


def test_caps_at_three():
    raw = "Q1?\nQ2?\nQ3?\nQ4?\nQ5?"
    result = _parse_sub_questions(raw, fallback="orig question")
    assert result == ["Q1?", "Q2?", "Q3?"]


def test_empty_response_falls_back_to_original_question():
    result = _parse_sub_questions("", fallback="orig question")
    assert result == ["orig question"]


def test_whitespace_only_response_falls_back():
    result = _parse_sub_questions("   \n\n   ", fallback="orig question")
    assert result == ["orig question"]


if __name__ == "__main__":
    test_parses_multiple_lines()
    test_strips_blank_lines()
    test_caps_at_three()
    test_empty_response_falls_back_to_original_question()
    test_whitespace_only_response_falls_back()
    print("=== all decompose parsing tests PASSED ===")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/apple/Desktop/Projects/aws-rag && source venv/bin/activate && python tests/test_decompose.py`
Expected: `ModuleNotFoundError: No module named 'src.nodes.decompose'`

- [ ] **Step 3: Write `src/nodes/decompose.py`**

```python
from __future__ import annotations

try:
    from src.nodes.llm import chat
    from src.state import RAGState
except ImportError:
    from nodes.llm import chat  # type: ignore
    from state import RAGState  # type: ignore


_SYSTEM = """You break down a hard, multi-hop question about AI research papers \
(Adaptive-RAG, Self-RAG, Chain-of-Verification, and related work) into 2-3 focused \
sub-questions that can each be answered by a SINGLE, independent document search.

Rules:
- Produce 2 or 3 sub-questions, never 1 and never more than 3.
- Each sub-question must be self-contained and independently searchable.
- Output ONLY the sub-questions, one per line.
- No numbering, no bullets, no explanation, no preamble."""

_MAX_SUB_QUESTIONS = 3


def _parse_sub_questions(raw: str, *, fallback: str) -> list[str]:
    lines = [line.strip() for line in raw.strip().splitlines()]
    questions = [line for line in lines if line]
    if not questions:
        return [fallback]
    return questions[:_MAX_SUB_QUESTIONS]


def decompose(state: RAGState) -> RAGState:
    question = state["question"]
    raw = chat(_SYSTEM, question)
    sub_questions = _parse_sub_questions(raw, fallback=question)

    trace = list(state.get("trace", []))
    trace.append(f"decompose → {len(sub_questions)} sub-question(s)")
    return {"sub_questions": sub_questions, "trace": trace}


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else (
        "Compare how BERT and the Transformer paper each handle positional "
        "information, and explain any tradeoffs."
    )
    result = decompose({"question": q, "trace": []})
    print(f"Question: {q}\n")
    for i, sq in enumerate(result["sub_questions"], start=1):
        print(f"  {i}. {sq}")
    print(f"\nTrace: {result['trace']}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/apple/Desktop/Projects/aws-rag && source venv/bin/activate && python tests/test_decompose.py`
Expected: `=== all decompose parsing tests PASSED ===`

- [ ] **Step 5: Commit**

```bash
git add src/nodes/decompose.py tests/test_decompose.py
git commit -m "Add decompose node for Mode 3 sub-question generation"
```

---

### Task 2: `critique.py` — parsing + fallback logic (offline, no Groq)

**Files:**
- Create: `src/nodes/critique.py`
- Test: `tests/test_critique.py`

**Interfaces:**
- Consumes: `src.state.RAGState` fields `question: str`, `answer: str`,
  `retrieved: list[dict]` (each dict has `document_name: str`, `page_number: int`,
  `section_heading: str | None`, `text: str` — same shape `src/nodes/generate.py:20-27`
  formats). Also consumes `src.nodes.llm.chat` (signature:
  `chat(system: str, user: str, *, temperature: float | None = None, max_tokens: int | None = None) -> str`).
- Produces: `critique(state: RAGState) -> RAGState` returning
  `{"critique_clean": bool, "unsupported_claims": list[str], "trace": list[str]}`.
  Phase 5 (out of scope here) will branch its conditional edge on `critique_clean` and
  feed `unsupported_claims` back in as the next iteration's `sub_questions`.

This task duplicates a small `_format_evidence` helper (identical logic to
`src/nodes/generate.py:20-27`) locally in `critique.py`, per the spec's note that this
decision should be made at implementation time — duplicating one 8-line function is
simpler than introducing a shared module for a single reused helper.

- [ ] **Step 1: Write the failing test for parsing**

Create `tests/test_critique.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nodes.critique import _parse_critique


def test_parses_clean_verdict():
    raw = "VERDICT: clean"
    clean, claims = _parse_critique(raw)
    assert clean is True
    assert claims == []


def test_parses_unsupported_with_claims():
    raw = (
        "VERDICT: unsupported\n"
        "CLAIMS:\n"
        "- What dataset did Self-RAG use for evaluation?\n"
        "- What is the exact reflection token vocabulary size?\n"
    )
    clean, claims = _parse_critique(raw)
    assert clean is False
    assert claims == [
        "What dataset did Self-RAG use for evaluation?",
        "What is the exact reflection token vocabulary size?",
    ]


def test_unparseable_response_falls_back_to_clean():
    raw = "I think the answer looks fine overall, no clear verdict here."
    clean, claims = _parse_critique(raw)
    assert clean is True
    assert claims == []


def test_empty_response_falls_back_to_clean():
    clean, claims = _parse_critique("")
    assert clean is True
    assert claims == []


def test_unsupported_with_no_claim_lines_falls_back_to_clean():
    raw = "VERDICT: unsupported\nCLAIMS:\n"
    clean, claims = _parse_critique(raw)
    assert clean is True
    assert claims == []


if __name__ == "__main__":
    test_parses_clean_verdict()
    test_parses_unsupported_with_claims()
    test_unparseable_response_falls_back_to_clean()
    test_empty_response_falls_back_to_clean()
    test_unsupported_with_no_claim_lines_falls_back_to_clean()
    print("=== all critique parsing tests PASSED ===")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/apple/Desktop/Projects/aws-rag && source venv/bin/activate && python tests/test_critique.py`
Expected: `ModuleNotFoundError: No module named 'src.nodes.critique'`

- [ ] **Step 3: Write `src/nodes/critique.py`**

```python
from __future__ import annotations

import re

try:
    from src.nodes.llm import chat
    from src.state import RAGState
except ImportError:
    from nodes.llm import chat  # type: ignore
    from state import RAGState  # type: ignore


_SYSTEM = """You verify whether an answer is fully supported by the numbered evidence \
provided. Check EVERY claim in the answer against the evidence.

Respond in exactly this format:

If every claim is supported:
VERDICT: clean

If any claim is NOT supported by the evidence:
VERDICT: unsupported
CLAIMS:
- <the unsupported claim, restated as a short, self-contained search question>
- <another unsupported claim, restated as a short, self-contained search question>

Restate each unsupported claim as a search-friendly question, not a verbatim quote — \
these will be used as follow-up search queries. Output ONLY the verdict block, no \
other text."""


def _format_evidence(retrieved: list[dict]) -> str:
    lines = []
    for i, c in enumerate(retrieved, start=1):
        loc = f"{c['document_name']} p{c['page_number']}"
        if c.get("section_heading"):
            loc += f" · {c['section_heading']}"
        lines.append(f"[{i}] ({loc}) {c['text']}")
    return "\n\n".join(lines)


def _parse_critique(raw: str) -> tuple[bool, list[str]]:
    text = raw.strip()
    verdict_match = re.search(r"VERDICT:\s*(clean|unsupported)", text, re.IGNORECASE)
    if verdict_match is None:
        return True, []

    verdict = verdict_match.group(1).lower()
    if verdict == "clean":
        return True, []

    claims_section = text[verdict_match.end():]
    claims = [
        line.strip().lstrip("-").strip()
        for line in claims_section.splitlines()
        if line.strip().startswith("-")
    ]
    claims = [c for c in claims if c]

    if not claims:
        return True, []

    return False, claims


def critique(state: RAGState) -> RAGState:
    question = state["question"]
    answer = state.get("answer", "")
    retrieved = state.get("retrieved", [])

    trace = list(state.get("trace", []))

    if not retrieved or not answer:
        trace.append("critique → clean (no evidence/answer to check)")
        return {"critique_clean": True, "unsupported_claims": [], "trace": trace}

    evidence = _format_evidence(retrieved)
    user = (
        f"Question: {question}\n\nAnswer:\n{answer}\n\nEvidence:\n{evidence}\n\n"
        "Verdict:"
    )
    raw = chat(_SYSTEM, user)
    clean, claims = _parse_critique(raw)

    trace.append(
        "critique → clean" if clean else f"critique → unsupported ({len(claims)} claim(s))"
    )
    return {"critique_clean": clean, "unsupported_claims": claims, "trace": trace}


if __name__ == "__main__":
    state = {
        "question": "What optimizer and learning rate schedule does the Transformer use?",
        "answer": (
            "The Transformer uses the Adam optimizer [1] with a warmup-then-decay "
            "learning rate schedule [1], and was trained on exactly 8 V100 GPUs for "
            "12 days [2]."
        ),
        "retrieved": [
            {
                "chunk_id": "c1",
                "document_name": "attention_is_all_you_need.pdf",
                "page_number": 7,
                "section_heading": "Training",
                "text": (
                    "We used the Adam optimizer with beta_1=0.9, beta_2=0.98, and "
                    "varied the learning rate over the course of training with a "
                    "warmup followed by decay proportional to the inverse square "
                    "root of the step number."
                ),
            }
        ],
        "trace": [],
    }
    result = critique(state)
    print(f"Clean: {result['critique_clean']}")
    print(f"Unsupported claims: {result['unsupported_claims']}")
    print(f"Trace: {result['trace']}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/apple/Desktop/Projects/aws-rag && source venv/bin/activate && python tests/test_critique.py`
Expected: `=== all critique parsing tests PASSED ===`

- [ ] **Step 5: Commit**

```bash
git add src/nodes/critique.py tests/test_critique.py
git commit -m "Add critique node for Mode 3 evidence-grounding checks"
```

---

### Task 3: Live Groq spot-checks for both nodes

**Files:**
- None created/modified — this task runs the `__main__` blocks already written in
  Task 1 Step 3 and Task 2 Step 3 against the real Groq API and the real FAISS index
  built in Phase 2, then records results in `BUILD_LOG.md` and flips Phase 4's checkbox
  in `PLAN.md`.

**Interfaces:**
- Consumes: `decompose(state)` and `critique(state)` as built in Tasks 1–2;
  `GROQ_API_KEY` from the already-configured `.env` (confirmed working in Phase 3,
  BUILD_LOG entry 8).

This task requires a live Groq call, so it's a manual verification task, not a
TDD unit-test task — matching how Phase 3's live routing spot-check was done
(BUILD_LOG entry 8) and how Phase 2's real-PDF ingestion was verified (BUILD_LOG
entry 7).

- [ ] **Step 1: Live spot-check `decompose()` on a real hard question**

Run: `cd /Users/apple/Desktop/Projects/aws-rag && source venv/bin/activate && python -m src.nodes.decompose`

Expected: prints the hard-coded test question and 2-3 sub-questions that each look like
an independently searchable question relevant to comparing BERT and the Transformer's
positional encodings. Read the output and confirm this by eye — no automated assertion,
this is a qualitative check same as the Phase 3 router spot-check.

- [ ] **Step 2: Live spot-check `critique()` on a deliberately unsupported claim**

Run: `cd /Users/apple/Desktop/Projects/aws-rag && source venv/bin/activate && python -m src.nodes.critique`

Expected: the hard-coded test state includes one grounded claim (Adam optimizer +
warmup/decay schedule, supported by the evidence) and one fabricated claim ("exactly 8
V100 GPUs for 12 days" — not in the evidence). Confirm the output shows
`Clean: False` and the unsupported-claims list contains a claim about the GPU/training
time, not the optimizer.

- [ ] **Step 3: Spot-check `critique()` on a fully-grounded answer (should pass clean)**

Run:

```bash
cd /Users/apple/Desktop/Projects/aws-rag && source venv/bin/activate && python -c "
from src.nodes.critique import critique

state = {
    'question': 'What optimizer does the Transformer use?',
    'answer': 'The Transformer uses the Adam optimizer [1] with a warmup-then-decay learning rate schedule [1].',
    'retrieved': [
        {
            'chunk_id': 'c1',
            'document_name': 'attention_is_all_you_need.pdf',
            'page_number': 7,
            'section_heading': 'Training',
            'text': 'We used the Adam optimizer with beta_1=0.9, beta_2=0.98, and varied the learning rate over the course of training with a warmup followed by decay proportional to the inverse square root of the step number.',
        }
    ],
    'trace': [],
}
result = critique(state)
print('Clean:', result['critique_clean'])
print('Unsupported claims:', result['unsupported_claims'])
"
```

Expected: `Clean: True` and `Unsupported claims: []`.

- [ ] **Step 4: Update `PLAN.md`**

Read `PLAN.md`, find the `## Phase 4 — Nodes, part B: Mode 3 building blocks  ☐`
heading and its body (currently 2 bullets: decompose.py and critique.py, plus an
isolation-test bullet and exit-check bullet). Change the heading checkbox from `☐` to
`☑`, and add the spot-check results as new content under it — following the same
style as the Phase 2/3 updates already in the file (status line + concrete results +
any known limitations).

- [ ] **Step 5: Add a `BUILD_LOG.md` entry**

Append a new entry (next sequential number after entry 8) documenting: what was built
(`decompose.py`, `critique.py`), why (Phase 4 per PLAN.md, unblocks Phase 5's graph
assembly), the offline unit-test results (Tasks 1–2, all passed), the live spot-check
results (Task 3, exact sub-questions / claims observed), and any deviations (none
expected — flag any that come up during implementation).

- [ ] **Step 6: Commit**

```bash
git add PLAN.md BUILD_LOG.md
git commit -m "Complete Phase 4: decompose + critique nodes verified live"
```

---

## Self-Review Notes

- **Spec coverage:** decompose.py (2-3 sub-questions, newline parsing, cap at 3,
  fallback to original question) — Task 1. critique.py (question+answer+evidence input,
  sentinel-line VERDICT/CLAIMS format, fallback to clean, claims phrased as search
  queries) — Task 2. Retry-wiring context is documentation-only for Phase 5, no task
  needed now. Live spot-checks — Task 3. All spec sections covered.
- **Placeholder scan:** no TBD/TODO; every step has complete runnable code; commands
  have concrete expected output.
- **Type consistency:** `decompose(state: RAGState) -> RAGState` and
  `critique(state: RAGState) -> RAGState` match the exact signature style of
  `route_question(state: RAGState) -> RAGState` (router.py) and
  `generate(state: RAGState) -> RAGState` (generate.py). `_parse_sub_questions` and
  `_parse_critique` names are used consistently between their definition (Tasks 1–2
  Step 3) and their tests (Tasks 1–2 Step 1).
