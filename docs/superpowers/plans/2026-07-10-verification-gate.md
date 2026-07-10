# Verification Gate, Pool Control, and Routing Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Mode-3 self-healing actually heal (catch wrong, fabricated, non-responsive, and shallow answers), stop single-doc questions escalating to hard, and stop the evidence pool poisoning generation.

**Architecture:** Three cooperating changes per the approved spec (`docs/superpowers/specs/2026-07-10-verification-gate-design.md`): (1) a deterministic routing guard downgrading hard→medium when exactly one indexed document is implicated by filename stem; (2) evidence-pool capping with a per-document quota applied at consumption time; (3) a `verify` node replacing `critique` — deterministic Stage 1 (citations, number grounding, coverage-via-embeddings) then binary LLM Stage 2 (responsiveness, support) — with failure-type-aware healing (regenerate vs re-search vs honest exit).

**Tech Stack:** Python, LangGraph, Ollama (qwen2.5:7b) via `src/nodes/llm.py chat()`, sentence-transformers MiniLM via `src/ingest.py embed_texts()`, FAISS, pytest.

## Global Constraints

- No emoji and no em-dashes in any user-visible string (project constraint).
- All LLM calls go through `src.nodes.llm.chat(system, user, max_tokens=..., label=...)`.
- Offline tests must not require Ollama or network: monkeypatch `chat` and `embed_texts`.
- State merging is plain TypedDict overwrite: every node returns complete values for fields it touches.
- Config values (spec §7): `POOL_CAP = 12`, `PER_DOC_CAP = 4`, `COVERAGE_SIM_THRESHOLD = 0.45`, `MEDIUM_ITERATION_CAP = 2`; `MAX_ITERATIONS` stays 3.
- Run offline suite with: `pytest tests/ --ignore=tests/test_questions.py -q` (all must pass at the end of every task).
- Each `src/nodes/*.py` and `src/*.py` file keeps the existing `try: from src... except ImportError:` import pattern.

---

### Task 1: `implicated_documents` in ingest.py

**Files:**
- Modify: `src/ingest.py` (add function after `match_document`, ~line 424)
- Test: `tests/test_ingest_match_document.py` (append tests)

**Interfaces:**
- Consumes: existing `_document_match_corpus()`, `_tokenize()` in `src/ingest.py`.
- Produces: `implicated_documents(text: str) -> list[str]` — sorted list of document names (e.g. `["bert.pdf", "roberta.pdf"]`) whose filename-stem tokens overlap the text's tokens. Stem-only on purpose: title tokens are too noisy ("the original Transformer" hits t5.pdf's title). Task 2 (router) depends on this exact signature.

- [ ] **Step 1: Write the failing tests**

Open `tests/test_ingest_match_document.py`, look at how existing tests monkeypatch or construct the corpus (they exercise `match_document` against the real or a stubbed index). Follow the same pattern the file already uses for corpus setup. Add:

```python
def test_implicated_documents_two_named_docs(monkeypatch):
    from src import ingest
    monkeypatch.setattr(
        ingest, "_document_match_corpus",
        lambda: {
            "bert.pdf": "BERT: Pre-training of Deep Bidirectional Transformers bert",
            "roberta.pdf": "RoBERTa: A Robustly Optimized BERT Pretraining Approach roberta",
            "t5.pdf": "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer t5",
        },
    )
    got = ingest.implicated_documents(
        "How does RoBERTa's pretraining differ from BERT's?"
    )
    assert got == ["bert.pdf", "roberta.pdf"]


def test_implicated_documents_stem_only_ignores_title_words(monkeypatch):
    from src import ingest
    monkeypatch.setattr(
        ingest, "_document_match_corpus",
        lambda: {
            "attention_is_all_you_need.pdf": "Attention Is All You Need attention_is_all_you_need",
            "t5.pdf": "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer t5",
        },
    )
    # "Transformer" appears in t5's TITLE but matches no filename stem: nothing implicated.
    got = ingest.implicated_documents(
        "What optimizer does the original Transformer use?"
    )
    assert got == []


def test_implicated_documents_dehyphenated_model_name(monkeypatch):
    from src import ingest
    monkeypatch.setattr(
        ingest, "_document_match_corpus",
        lambda: {
            "gpt2.pdf": "Language Models are Unsupervised Multitask Learners gpt2",
            "gpt3.pdf": "Language Models are Few-Shot Learners gpt3",
        },
    )
    got = ingest.implicated_documents("Trace the evolution from GPT-2 to GPT-3")
    assert got == ["gpt2.pdf", "gpt3.pdf"]


def test_implicated_documents_empty_corpus(monkeypatch):
    from src import ingest
    monkeypatch.setattr(ingest, "_document_match_corpus", lambda: {})
    assert ingest.implicated_documents("anything") == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ingest_match_document.py -q -k implicated`
Expected: 4 FAIL with `AttributeError: module 'src.ingest' has no attribute 'implicated_documents'`

- [ ] **Step 3: Implement**

Add to `src/ingest.py` directly after `match_document` (after line 423):

```python
def implicated_documents(text: str) -> list[str]:
    """Every indexed document whose FILENAME-STEM tokens overlap the text's
    tokens. Stem-only (never title words): titles share vocabulary across this
    corpus ("Transformer" is in t5's title), so title matching would implicate
    the wrong document. A stem hit ("bert", "gpt2" via de-hyphenation) is the
    unambiguous signal that the user named that document. Used by the router's
    hard->medium downgrade guard."""
    corpus = _document_match_corpus()
    if not corpus:
        return []
    query_tokens = _tokenize(text)
    implicated = []
    for name in corpus:
        stem_tokens = _tokenize(Path(name).stem.replace("_", " "))
        if query_tokens & stem_tokens:
            implicated.append(name)
    return sorted(implicated)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ingest_match_document.py -q`
Expected: all PASS (new 4 plus every pre-existing test in the file)

- [ ] **Step 5: Commit**

```bash
git add src/ingest.py tests/test_ingest_match_document.py
git commit -m "feat: add implicated_documents stem-only multi-doc matcher"
```

---

### Task 2: Routing guard (hard→medium downgrade) in router.py

**Files:**
- Modify: `src/nodes/router.py` (import + logic after the easy→medium upgrade block, lines ~110-118)
- Test: `tests/test_router.py` (append tests)

**Interfaces:**
- Consumes: `implicated_documents(text) -> list[str]` from Task 1.
- Produces: routing behavior only (no new API). Trace line format other code may read: `router → hard downgraded to medium (implicates only <doc>)`.

- [ ] **Step 1: Write the failing tests**

Open `tests/test_router.py` and follow its existing monkeypatch pattern for stubbing `chat` and `match_document`. Add (adjust the import/monkeypatch style to match the file's existing tests exactly):

```python
def test_hard_downgraded_to_medium_when_one_doc_implicated(monkeypatch):
    from src.nodes import router
    monkeypatch.setattr(router, "chat", lambda *a, **k: "hard")
    monkeypatch.setattr(router, "match_document", lambda q: None)
    monkeypatch.setattr(
        router, "implicated_documents", lambda q: ["palm.pdf"]
    )
    result = router.route_question({"question": "What is PaLM's parameter count and what scaling insight did it demonstrate?", "trace": []})
    assert result["route"] == "medium"
    assert any("downgraded to medium" in t for t in result["trace"])


def test_hard_stays_hard_with_two_docs_implicated(monkeypatch):
    from src.nodes import router
    monkeypatch.setattr(router, "chat", lambda *a, **k: "hard")
    monkeypatch.setattr(router, "match_document", lambda q: None)
    monkeypatch.setattr(
        router, "implicated_documents", lambda q: ["bert.pdf", "roberta.pdf"]
    )
    result = router.route_question({"question": "How does RoBERTa differ from BERT?", "trace": []})
    assert result["route"] == "hard"


def test_hard_stays_hard_with_zero_docs_implicated(monkeypatch):
    from src.nodes import router
    monkeypatch.setattr(router, "chat", lambda *a, **k: "hard")
    monkeypatch.setattr(router, "match_document", lambda q: None)
    monkeypatch.setattr(router, "implicated_documents", lambda q: [])
    result = router.route_question({"question": "What themes recur across the corpus?", "trace": []})
    assert result["route"] == "hard"


def test_medium_not_upgraded_by_guard(monkeypatch):
    from src.nodes import router
    monkeypatch.setattr(router, "chat", lambda *a, **k: "medium")
    monkeypatch.setattr(router, "match_document", lambda q: None)
    monkeypatch.setattr(
        router, "implicated_documents", lambda q: ["bert.pdf", "roberta.pdf"]
    )
    result = router.route_question({"question": "anything", "trace": []})
    assert result["route"] == "medium"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_router.py -q`
Expected: the 4 new tests FAIL — first with `AttributeError` (router has no `implicated_documents` attribute to monkeypatch), the rest with `assert 'hard' == 'medium'` style errors. Pre-existing tests still pass.

- [ ] **Step 3: Implement**

In `src/nodes/router.py`:

1. Change both import blocks to also import `implicated_documents`:

```python
    from src.ingest import match_document, implicated_documents
```
(and in the `except ImportError` branch: `from ingest import match_document, implicated_documents  # type: ignore`)

2. After the existing easy→medium upgrade block (after line 117, before the final `log.info`), add:

```python
    # Deterministic downgrade guard: a question that names exactly ONE indexed
    # document is single-hop by definition (one scoped search surfaces its
    # evidence, even for two-part questions), so the hard path's latency and
    # pool growth are pure cost. 0 implicated docs stays hard: corpus-wide
    # synthesis, or a paraphrase we cannot scope (safe, slower-not-wrong).
    if label == config.ROUTE_HARD:
        implicated = implicated_documents(question)
        if len(implicated) == 1:
            label = config.ROUTE_MEDIUM
            log.info(
                "Q=%r route=hard downgraded to medium (implicates only %s)",
                question[:70], implicated[0],
            )
            trace.append(
                f"router → hard downgraded to medium (implicates only {implicated[0]})"
            )
```

3. Also reset the new healing fields per run. Change the final return to:

```python
    return {
        "route": label,
        "mode_label": config.MODE_LABELS[label],
        "iterations": 0,
        "failure_type": "",
        "verify_feedback": "",
        "retry_queries": [],
        "verification_warnings": [],
        "trace": trace,
    }
```

And add the same four reset fields to the early `meta` return dict (the one returning `ROUTE_META`).

Note: these state fields are declared in `src/state.py` in Task 4; TypedDict is not enforced at runtime, so returning them now is safe and keeps this task self-contained.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_router.py -q`
Expected: all PASS (new and pre-existing).

- [ ] **Step 5: Commit**

```bash
git add src/nodes/router.py tests/test_router.py
git commit -m "feat: deterministic hard->medium routing guard via implicated_documents"
```

---

### Task 3: Pool capping with per-document quota + retry-query consumption in search.py

**Files:**
- Modify: `src/config.py` (add `POOL_CAP`, `PER_DOC_CAP`)
- Modify: `src/nodes/search.py` (add `cap_pool`, consume `retry_queries`)
- Test: `tests/test_search.py` (append tests)

**Interfaces:**
- Consumes: `state["retry_queries"]` (set by `prepare_research` in Task 6; empty list otherwise).
- Produces: `cap_pool(pooled: list[dict]) -> list[dict]` — public, deterministic; generate (Task 5) and verify (Task 4) call it so both see the identical capped evidence list with identical ordering (citation indices must line up). `search_node` continues to store the FULL merged pool in `state["retrieved"]` and keeps incrementing `iterations`; it returns `retry_queries: []` after consuming them.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_search.py` (match the file's existing import style):

```python
def test_cap_pool_respects_total_and_per_doc_caps():
    from src.nodes.search import cap_pool
    # 14 chunks from docA (scores 14.0..1.0), 3 from docB (0.9..0.7):
    # the pool EXCEEDS POOL_CAP, so the quota+backfill actually bite. Plain
    # top-12 by score would be 12 docA chunks and zero docB; the quota pass
    # takes 4a+3b, backfill fills the rest with the best skipped a-chunks.
    pool = [
        {"chunk_id": f"a{i}", "document_name": "a.pdf", "score": float(14 - i)}
        for i in range(14)
    ] + [
        {"chunk_id": f"b{i}", "document_name": "b.pdf", "score": 0.9 - i * 0.1}
        for i in range(3)
    ]
    capped = cap_pool(pool)
    a_count = sum(1 for c in capped if c["document_name"] == "a.pdf")
    b_count = sum(1 for c in capped if c["document_name"] == "b.pdf")
    assert b_count == 3   # docB's evidence survives (the F2 protection)
    assert a_count == 9   # 4 quota + 5 backfilled
    assert len(capped) == 12  # POOL_CAP


def test_cap_pool_backfills_when_quota_leaves_room():
    from src.nodes.search import cap_pool
    from src import config
    # One doc with 20 chunks: quota selects 4, backfill tops the pool to POOL_CAP.
    pool = [
        {"chunk_id": f"a{i}", "document_name": "a.pdf", "score": float(20 - i)}
        for i in range(20)
    ]
    capped = cap_pool(pool)
    assert len(capped) == config.POOL_CAP
    # Highest-scored chunks win the backfill
    assert capped[0]["chunk_id"] == "a0"


def test_cap_pool_under_cap_unchanged():
    from src.nodes.search import cap_pool
    pool = [{"chunk_id": "x", "document_name": "a.pdf", "score": 1.0}]
    assert cap_pool(pool) == pool


def test_search_node_consumes_retry_queries(monkeypatch):
    from src.nodes import search as search_mod
    calls = []

    def fake_search(q, k=None, document_hint=None):
        calls.append(q)
        return []

    monkeypatch.setattr(search_mod, "faiss_search", fake_search)
    monkeypatch.setattr(search_mod, "match_document", lambda q: None)
    state = {
        "question": "original question",
        "sub_questions": ["sub q one", "sub q two"],
        "retry_queries": ["retry query"],
        "retrieved": [],
        "iterations": 1,
        "trace": [],
    }
    result = search_mod.search_node(state)
    assert calls == ["retry query"]          # retry queries win over sub_questions
    assert result["retry_queries"] == []      # consumed, cleared
    assert result["iterations"] == 2          # still increments
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_search.py -q`
Expected: new tests FAIL (`ImportError: cannot import name 'cap_pool'`; retry-queries assertion fails because sub_questions are used).

- [ ] **Step 3: Implement**

In `src/config.py`, after `TOP_K = 6`:

```python
# Evidence-pool control: generate/verify consume at most POOL_CAP chunks, and
# no single document may crowd out another's evidence (protects N-way
# comparison questions from having one paper's chunks evicted entirely).
POOL_CAP = 12
PER_DOC_CAP = 4
```

In `src/nodes/search.py`:

1. Add after `_merge` (line 23):

```python
def cap_pool(pooled: list[dict]) -> list[dict]:
    """Deterministically cap a score-sorted pool to POOL_CAP chunks with at
    most PER_DOC_CAP per document (quota pass), backfilling from the skipped
    chunks if the quota leaves room. generate and verify both call this on
    state["retrieved"] so they see the identical evidence list: citation
    indices in the answer must resolve against the same numbering."""
    selected: list[dict] = []
    skipped: list[dict] = []
    per_doc: dict[str, int] = {}
    for c in pooled:
        doc = c["document_name"]
        if per_doc.get(doc, 0) < config.PER_DOC_CAP:
            selected.append(c)
            per_doc[doc] = per_doc.get(doc, 0) + 1
        else:
            skipped.append(c)
        if len(selected) >= config.POOL_CAP:
            break
    if len(selected) < config.POOL_CAP and skipped:
        selected.extend(skipped[: config.POOL_CAP - len(selected)])
        selected.sort(key=lambda c: c.get("score", 0.0), reverse=True)
    return selected
```

2. In `search_node`, replace the queries line (line 27-28):

```python
    sub_questions = state.get("sub_questions") or []
    retry_queries = state.get("retry_queries") or []
    # Re-search retries (from verify's support failures) take priority; they
    # never overwrite sub_questions, which the coverage check still needs.
    queries = retry_queries or sub_questions or [state["question"]]
```

3. Add `"retry_queries": []` to `search_node`'s return dict (consumed exactly once).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_search.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/config.py src/nodes/search.py tests/test_search.py
git commit -m "feat: evidence pool capping with per-doc quota and retry-query consumption"
```

---

### Task 4: The verify node (replaces critique)

**Files:**
- Create: `src/nodes/verify.py`
- Modify: `src/state.py` (new fields)
- Modify: `src/config.py` (add `COVERAGE_SIM_THRESHOLD`, `MEDIUM_ITERATION_CAP`)
- Delete: `src/nodes/critique.py`, `tests/test_critique.py` (support prompt/parser absorbed into verify; parser tests migrated)
- Test: `tests/test_verify.py` (new)

**Interfaces:**
- Consumes: `chat(system, user, max_tokens=..., label=...)` from `src.nodes.llm`; `embed_texts(texts) -> np.ndarray` from `src.ingest`; `cap_pool` from Task 3.
- Produces: `verify(state: RAGState) -> RAGState` returning COMPLETE values for: `critique_clean: bool`, `failure_type: str` (one of `"", "citations", "fabrication", "coverage", "responsiveness", "support"`), `verify_feedback: str`, `unsupported_claims: list[str]`, `heal_action: str` (one of `"none", "regenerate", "research"`), `verification_warnings: list[str]`, `iterations: int`, `answer: str` (may append an honest-exit note), `trace: list[str]`. Task 6's graph edge reads `heal_action` only.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_verify.py`:

```python
"""Offline tests for the verification gate. chat and embed_texts are
monkeypatched; no Ollama or network needed."""
import numpy as np

from src.nodes import verify as verify_mod
from src.nodes.verify import (
    _canon_number,
    _check_citations,
    _check_numbers,
    _parse_support,
    verify,
)


def _evidence(texts):
    return [
        {
            "chunk_id": f"c{i}",
            "document_name": "doc.pdf",
            "page_number": 1,
            "section_heading": "",
            "text": t,
            "score": 1.0,
        }
        for i, t in enumerate(texts, start=1)
    ]


# ---------- number canonicalization ----------

def test_canon_number_folds_commas_and_magnitudes():
    assert _canon_number("4,000") == "4000"
    assert _canon_number("175B") == "175billion"
    assert _canon_number("175 billion") == "175billion"
    assert _canon_number("93.3%") == "93.3%"
    assert _canon_number("1/4") == "1/4"


# ---------- citation check ----------

def test_check_citations_ok():
    ok, fb = _check_citations("Adam is used [1] with warmup [2].", 3)
    assert ok

def test_check_citations_out_of_range():
    ok, fb = _check_citations("Adam is used [7].", 3)
    assert not ok

def test_check_citations_malformed_marker():
    ok, fb = _check_citations("PaLM outperforms [n14] and [n].", 3)
    assert not ok

def test_check_citations_none_present():
    ok, fb = _check_citations("An answer with no citations at all.", 3)
    assert not ok


# ---------- number grounding ----------

def test_check_numbers_fabricated_value_fails():
    evid = "ELECTRA performs comparably while using less than 1/4 of their compute."
    ok, offending = _check_numbers(
        "ELECTRA uses 1/135 of the compute [1].", evid, question=""
    )
    assert not ok
    assert "1/135" in offending[0]

def test_check_numbers_normalized_match_passes():
    evid = "We used warmup_steps = 4000 with 175 billion parameters."
    ok, offending = _check_numbers(
        "It uses 4,000 warmup steps [1] and 175B parameters [1].", evid, question=""
    )
    assert ok

def test_check_numbers_question_numbers_allowed():
    ok, offending = _check_numbers(
        "GPT-3 has more parameters [1].", "the model is larger", question="What about GPT-3?"
    )
    assert ok  # "3" comes from the question, not fabrication


# ---------- support parser (migrated from critique) ----------

def test_parse_support_clean():
    clean, claims = _parse_support("VERDICT: clean")
    assert clean and claims == []

def test_parse_support_unsupported_with_claims():
    raw = "VERDICT: unsupported\nCLAIMS:\n- What optimizer is used?\n- What is the warmup?"
    clean, claims = _parse_support(raw)
    assert not clean
    assert claims == ["What optimizer is used?", "What is the warmup?"]

def test_parse_support_garbage_defaults_clean():
    clean, claims = _parse_support("no verdict here")
    assert clean


# ---------- verify orchestration ----------

def _patch_embeddings(monkeypatch, sim):
    """Make every sub-question/sentence pair have cosine `sim`."""
    def fake_embed(texts):
        v = np.zeros((len(texts), 3), dtype="float32")
        v[:, 0] = 1.0 if sim >= 0.99 else 0.0
        v[:, 1] = 0.0 if sim >= 0.99 else 1.0
        # sub-questions and sentences get identical vectors when sim high,
        # orthogonal when low; caller embeds sub-qs and sentences separately.
        return v
    monkeypatch.setattr(verify_mod, "embed_texts", fake_embed)


def test_verify_fabrication_dispatches_regenerate(monkeypatch):
    monkeypatch.setattr(verify_mod, "chat", lambda *a, **k: "yes")
    _patch_embeddings(monkeypatch, sim=1.0)
    state = {
        "question": "How efficient is ELECTRA?",
        "route": "hard",
        "answer": "ELECTRA uses 1/135 of the compute [1].",
        "retrieved": _evidence(["ELECTRA uses less than 1/4 of their compute."]),
        "sub_questions": ["How efficient is ELECTRA?"],
        "iterations": 1,
        "trace": [],
    }
    r = verify(state)
    assert r["failure_type"] == "fabrication"
    assert r["heal_action"] == "regenerate"
    assert r["iterations"] == 2          # regenerate dispatch pre-increments
    assert not r["critique_clean"]
    assert "1/135" in r["verify_feedback"]


def test_verify_support_failure_dispatches_research(monkeypatch):
    def fake_chat(system, user, **kw):
        if kw.get("label") == "verify-responsiveness":
            return "yes"
        return "VERDICT: unsupported\nCLAIMS:\n- What optimizer does it use?"
    monkeypatch.setattr(verify_mod, "chat", fake_chat)
    _patch_embeddings(monkeypatch, sim=1.0)
    state = {
        "question": "What optimizer?",
        "route": "hard",
        "answer": "It uses Adam [1].",
        "retrieved": _evidence(["Adam optimizer text."]),
        "sub_questions": ["What optimizer?"],
        "iterations": 1,
        "trace": [],
    }
    r = verify(state)
    assert r["failure_type"] == "support"
    assert r["heal_action"] == "research"
    assert r["iterations"] == 1          # search will do the incrementing
    assert r["unsupported_claims"] == ["What optimizer does it use?"]


def test_verify_responsiveness_failure(monkeypatch):
    def fake_chat(system, user, **kw):
        if kw.get("label") == "verify-responsiveness":
            return "no"
        return "VERDICT: clean"
    monkeypatch.setattr(verify_mod, "chat", fake_chat)
    _patch_embeddings(monkeypatch, sim=1.0)
    state = {
        "question": "What optimizer does the Transformer use?",
        "route": "hard",
        "answer": "Self-attention processes sequences [1].",
        "retrieved": _evidence(["Self-attention text about sequences."]),
        "sub_questions": ["What optimizer does the Transformer use?"],
        "iterations": 1,
        "trace": [],
    }
    r = verify(state)
    assert r["failure_type"] == "responsiveness"
    assert r["heal_action"] == "regenerate"


def test_verify_budget_exhausted_honest_exit(monkeypatch):
    monkeypatch.setattr(verify_mod, "chat", lambda *a, **k: "yes")
    _patch_embeddings(monkeypatch, sim=1.0)
    state = {
        "question": "How efficient?",
        "route": "hard",
        "answer": "It uses 1/135 of the compute [1].",
        "retrieved": _evidence(["less than 1/4 of compute."]),
        "sub_questions": ["How efficient?"],
        "iterations": 3,                  # budget already spent
        "trace": [],
    }
    r = verify(state)
    assert r["heal_action"] == "none"
    assert r["verification_warnings"]     # surfaced, not silent
    assert "could not be fully verified" in r["answer"]


def test_verify_medium_skips_stage2_and_caps_at_one_heal(monkeypatch):
    called = {"chat": 0}
    def fake_chat(*a, **k):
        called["chat"] += 1
        return "yes"
    monkeypatch.setattr(verify_mod, "chat", fake_chat)
    state = {
        "question": "What is X?",
        "route": "medium",
        "answer": "X is Y [1].",
        "retrieved": _evidence(["X is Y."]),
        "sub_questions": [],
        "iterations": 1,
        "trace": [],
    }
    r = verify(state)
    assert r["critique_clean"] and r["heal_action"] == "none"
    assert called["chat"] == 0            # medium: Stage 1 only, no LLM

    # medium with a fabricated number at iterations=2: budget (2) exhausted
    state2 = dict(state, answer="X is 999 [1].", iterations=2)
    r2 = verify(state2)
    assert r2["heal_action"] == "none"
    assert r2["verification_warnings"]


def test_verify_clean_passes_everything(monkeypatch):
    def fake_chat(system, user, **kw):
        if kw.get("label") == "verify-responsiveness":
            return "yes"
        return "VERDICT: clean"
    monkeypatch.setattr(verify_mod, "chat", fake_chat)
    _patch_embeddings(monkeypatch, sim=1.0)
    state = {
        "question": "What optimizer?",
        "route": "hard",
        "answer": "Adam with 4000 warmup steps [1].",
        "retrieved": _evidence(["We used the Adam optimizer with warmup_steps = 4000."]),
        "sub_questions": ["What optimizer?"],
        "iterations": 1,
        "trace": [],
    }
    r = verify(state)
    assert r["critique_clean"] is True
    assert r["failure_type"] == ""
    assert r["heal_action"] == "none"
    assert r["verification_warnings"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_verify.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.nodes.verify'`

- [ ] **Step 3: Implement `src/nodes/verify.py`**

```python
from __future__ import annotations

import re

import numpy as np

try:
    from src import config
    from src.ingest import embed_texts
    from src.logging_config import get_logger
    from src.nodes.llm import chat
    from src.nodes.search import cap_pool
    from src.state import RAGState
except ImportError:
    import config  # type: ignore
    from ingest import embed_texts  # type: ignore
    from logging_config import get_logger  # type: ignore
    from nodes.llm import chat  # type: ignore
    from nodes.search import cap_pool  # type: ignore
    from state import RAGState  # type: ignore

log = get_logger("verify")

# Failure types dispatched to regeneration (evidence is present; the prose is
# wrong) versus re-search (evidence is genuinely missing).
_REGEN_FAILURES = {"citations", "fabrication", "coverage", "responsiveness"}

_MARKER_RE = re.compile(r"\[n?\d*\]")
_VALID_MARKER_RE = re.compile(r"\[(\d+)\]")
_MALFORMED_MARKER_RE = re.compile(r"\[n\d*\]")

_NUM_RE = re.compile(
    r"\d+/\d+"                                   # fractions: 1/4, 1/135
    r"|\d+(?:,\d{3})*(?:\.\d+)?"                 # 4,000  93.3  175
    r"(?:\s*(?:%|percent|billion|million|thousand|[bmk]\b))?",
    re.IGNORECASE,
)

_MAGNITUDES = {"b": "billion", "m": "million", "k": "thousand", "percent": "%"}


def _canon_number(tok: str) -> str:
    t = tok.lower().replace(",", "").strip()
    t = re.sub(r"\s+", "", t)
    m = re.match(r"^(\d+(?:\.\d+)?)(%|percent|billion|million|thousand|b|m|k)$", t)
    if m:
        value, suffix = m.group(1), m.group(2)
        suffix = _MAGNITUDES.get(suffix, suffix)
        return f"{value}{suffix}"
    return t


def _extract_numbers(text: str) -> list[str]:
    return [m.group(0) for m in _NUM_RE.finditer(text)]


def _check_citations(answer: str, n_evidence: int) -> tuple[bool, str]:
    if _MALFORMED_MARKER_RE.search(answer) or re.search(r"\[n\]", answer):
        return False, "the answer contains malformed citation markers like [n]"
    markers = [int(m) for m in _VALID_MARKER_RE.findall(answer)]
    if not markers:
        return False, "the answer contains no [n] citations"
    bad = sorted({m for m in markers if not (1 <= m <= n_evidence)})
    if bad:
        return False, f"citation markers {bad} do not match any evidence (1..{n_evidence})"
    return True, ""


def _check_numbers(answer: str, evidence_text: str, question: str) -> tuple[bool, list[str]]:
    body = _MARKER_RE.sub(" ", answer)  # citation indices are not claims
    allowed = {
        _canon_number(t)
        for t in _extract_numbers(evidence_text) + _extract_numbers(question)
    }
    evidence_canon = re.sub(r"[,\s]", "", evidence_text.lower())
    offending = []
    for tok in _extract_numbers(body):
        canon = _canon_number(tok)
        if canon in allowed:
            continue
        # Boundary-anchored fallback: the canonical value must appear in the
        # evidence as a whole number, not as a substring of a larger one
        # ("400" must NOT pass because "14000" exists). Reject digit/decimal
        # continuation on either side; a sentence-final period still matches.
        pattern = r"(?<![\d.])" + re.escape(canon) + r"(?!\.?\d)"
        if re.search(pattern, evidence_canon):
            continue
        offending.append(tok)
    return (not offending), offending


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _check_coverage(sub_questions: list[str], answer: str) -> tuple[bool, list[str]]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(answer) if len(s.strip()) > 20]
    if not sub_questions or not sentences:
        return True, []
    q_vecs = np.asarray(embed_texts(list(sub_questions)), dtype="float32")
    s_vecs = np.asarray(embed_texts(sentences), dtype="float32")

    def _norm(v):
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return v / n

    sims = _norm(q_vecs) @ _norm(s_vecs).T  # (n_subq, n_sentences)
    missed = [
        sq
        for sq, row in zip(sub_questions, sims)
        if float(row.max()) < config.COVERAGE_SIM_THRESHOLD
    ]
    return (not missed), missed


_RESPONSIVENESS_SYSTEM = (
    "You judge whether an answer directly addresses a question. "
    "Reply with exactly one word: yes or no."
)


def _check_responsiveness(question: str, answer: str) -> bool:
    raw = chat(
        _RESPONSIVENESS_SYSTEM,
        f"Question: {question}\n\nAnswer:\n{answer}\n\n"
        "Does the answer directly address what the question asked?",
        max_tokens=8,
        label="verify-responsiveness",
    )
    # Fail-open on garbage: the support check still runs, and looping on an
    # unparseable verdict would burn the heal budget for nothing.
    return not raw.strip().lower().startswith("no")


# Support check: absorbed from the former critique node, unchanged behavior.
_SUPPORT_SYSTEM = """You verify whether an answer is fully supported by the numbered evidence \
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


def _parse_support(raw: str) -> tuple[bool, list[str]]:
    text = raw.strip()
    verdict_match = re.search(r"VERDICT:\s*(clean|unsupported)", text, re.IGNORECASE)
    if verdict_match is None:
        return True, []
    if verdict_match.group(1).lower() == "clean":
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


def _format_evidence(retrieved: list[dict]) -> str:
    lines = []
    for i, c in enumerate(retrieved, start=1):
        loc = f"{c['document_name']} p{c['page_number']}"
        if c.get("section_heading"):
            loc += f" · {c['section_heading']}"
        lines.append(f"[{i}] ({loc}) {c['text']}")
    return "\n\n".join(lines)


def _check_support(question: str, answer: str, evidence: str) -> tuple[bool, list[str]]:
    raw = chat(
        _SUPPORT_SYSTEM,
        f"Question: {question}\n\nAnswer:\n{answer}\n\nEvidence:\n{evidence}\n\nVerdict:",
        label="verify-support",
    )
    return _parse_support(raw)


def verify(state: RAGState) -> RAGState:
    question = state["question"]
    answer = state.get("answer", "")
    route = state.get("route", "")
    sub_questions = state.get("sub_questions") or []
    iterations = state.get("iterations", 0)
    trace = list(state.get("trace", []))

    pool = cap_pool(state.get("retrieved", []))
    if not pool or not answer:
        trace.append("verify → clean (no evidence/answer to check)")
        return {
            "critique_clean": True, "failure_type": "", "verify_feedback": "",
            "unsupported_claims": [], "heal_action": "none",
            "verification_warnings": [], "iterations": iterations,
            "answer": answer, "trace": trace,
        }

    evidence_text = " ".join(c["text"] for c in pool)
    failure_type = ""
    feedback = ""
    unsupported: list[str] = []

    # ---- Stage 1: deterministic, zero LLM calls ----
    ok, fb = _check_citations(answer, len(pool))
    if not ok:
        failure_type, feedback = "citations", fb
    if not failure_type:
        ok, offending = _check_numbers(answer, evidence_text, question)
        if not ok:
            failure_type = "fabrication"
            feedback = (
                "the answer contains values not present in the evidence: "
                + ", ".join(offending)
            )
    if not failure_type and route == config.ROUTE_HARD:
        ok, missed = _check_coverage(sub_questions, answer)
        if not ok:
            failure_type = "coverage"
            feedback = "the answer does not address: " + "; ".join(missed)

    # ---- Stage 2: binary LLM checks (hard route only, only if Stage 1 passed) ----
    if not failure_type and route == config.ROUTE_HARD:
        if not _check_responsiveness(question, answer):
            failure_type = "responsiveness"
            feedback = "the answer does not directly address the question"
        else:
            clean, claims = _check_support(question, answer, _format_evidence(pool))
            if not clean:
                failure_type = "support"
                unsupported = claims
                feedback = "claims lacking evidence: " + "; ".join(claims)

    if not failure_type:
        log.info("verdict=clean")
        trace.append("verify → clean")
        return {
            "critique_clean": True, "failure_type": "", "verify_feedback": "",
            "unsupported_claims": [], "heal_action": "none",
            "verification_warnings": [], "iterations": iterations,
            "answer": answer, "trace": trace,
        }

    # ---- Healing dispatch, bounded by the iteration budget ----
    budget = (
        config.MEDIUM_ITERATION_CAP
        if route == config.ROUTE_MEDIUM
        else config.MAX_ITERATIONS
    )
    if iterations >= budget:
        warning = f"{failure_type}: {feedback}"
        note = (
            "\n\nNote: this answer could not be fully verified against the "
            f"indexed documents ({feedback})."
        )
        log.info("verdict=%s, budget exhausted (%d/%d) -> honest exit",
                 failure_type, iterations, budget)
        trace.append(f"verify → {failure_type}, budget exhausted, honest exit")
        return {
            "critique_clean": False, "failure_type": failure_type,
            "verify_feedback": feedback, "unsupported_claims": unsupported,
            "heal_action": "none", "verification_warnings": [warning],
            "iterations": iterations, "answer": answer + note, "trace": trace,
        }

    if failure_type in _REGEN_FAILURES:
        heal_action = "regenerate"
        iterations += 1  # regeneration bypasses search_node's counter
    else:
        heal_action = "research"  # search_node will increment

    log.info("verdict=%s -> %s (iteration %d)", failure_type, heal_action, iterations)
    trace.append(f"verify → {failure_type} → {heal_action}")
    return {
        "critique_clean": False, "failure_type": failure_type,
        "verify_feedback": feedback, "unsupported_claims": unsupported,
        "heal_action": heal_action, "verification_warnings": [],
        "iterations": iterations, "answer": answer, "trace": trace,
    }
```

- [ ] **Step 4: Update `src/state.py`**

Add to `RAGState` after `iterations: int`:

```python
    failure_type: str
    verify_feedback: str
    heal_action: str
    retry_queries: list[str]
    verification_warnings: list[str]
```

- [ ] **Step 5: Update `src/config.py`**

After `MAX_ITERATIONS = 3`:

```python
# Medium answers get one regeneration retry: initial search (1) + one heal (2).
MEDIUM_ITERATION_CAP = 2

# Coverage check: a decompose sub-question counts as addressed when some answer
# sentence reaches this cosine similarity (MiniLM embeddings). Tuned against
# the real corpus; too low misses shallow answers, too high forces needless
# regeneration.
COVERAGE_SIM_THRESHOLD = 0.45
```

- [ ] **Step 6: Delete critique module and migrate its tests**

```bash
git rm src/nodes/critique.py tests/test_critique.py
```

(The support prompt and parser now live in verify.py; `_parse_support` tests in test_verify.py replace the old parser tests. Check `tests/test_graph.py` for critique imports — it will be updated in Task 6; if it fails at this point, note it and proceed, Task 6 restores the suite.)

- [ ] **Step 7: Run tests**

Run: `pytest tests/test_verify.py -q`
Expected: all PASS.
Run: `pytest tests/ --ignore=tests/test_questions.py -q`
Expected: test_graph.py may FAIL on the critique import (fixed in Task 6); everything else PASS.

- [ ] **Step 8: Commit**

```bash
git add src/nodes/verify.py src/state.py src/config.py tests/test_verify.py
git commit -m "feat: verification gate node (deterministic checks + binary LLM critique), absorbing critique"
```

---

### Task 5: Regeneration feedback in generate.py

**Files:**
- Modify: `src/nodes/generate.py`
- Test: `tests/test_generate.py` (append tests)

**Interfaces:**
- Consumes: `state["failure_type"]`, `state["verify_feedback"]` (set by verify, Task 4); `cap_pool` from Task 3.
- Produces: generate consumes the CAPPED pool (identical list to verify's) and, when `verify_feedback` is set, appends a correction block to the user prompt.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_generate.py` (match its existing monkeypatch style for `chat`):

```python
def test_generate_uses_capped_pool(monkeypatch):
    from src.nodes import generate as gen_mod
    from src import config
    seen = {}
    def fake_chat(system, user, **kw):
        seen["user"] = user
        return "Answer [1]."
    monkeypatch.setattr(gen_mod, "chat", fake_chat)
    retrieved = [
        {"chunk_id": f"c{i}", "document_name": f"d{i % 6}.pdf", "page_number": 1,
         "section_heading": "", "text": f"chunk text {i}", "score": float(100 - i)}
        for i in range(30)
    ]
    result = gen_mod.generate({"question": "q?", "retrieved": retrieved, "trace": []})
    # Only POOL_CAP chunks appear in the prompt
    assert f"[{config.POOL_CAP}]" in seen["user"]
    assert f"[{config.POOL_CAP + 1}]" not in seen["user"]


def test_generate_appends_verify_feedback(monkeypatch):
    from src.nodes import generate as gen_mod
    seen = {}
    def fake_chat(system, user, **kw):
        seen["user"] = user
        return "Corrected answer [1]."
    monkeypatch.setattr(gen_mod, "chat", fake_chat)
    state = {
        "question": "q?",
        "retrieved": [{"chunk_id": "c1", "document_name": "d.pdf", "page_number": 1,
                       "section_heading": "", "text": "evidence", "score": 1.0}],
        "failure_type": "fabrication",
        "verify_feedback": "the answer contains values not present in the evidence: 1/135",
        "trace": [],
    }
    gen_mod.generate(state)
    assert "PREVIOUS ATTEMPT FAILED VERIFICATION" in seen["user"]
    assert "1/135" in seen["user"]


def test_generate_no_feedback_block_when_clean(monkeypatch):
    from src.nodes import generate as gen_mod
    seen = {}
    def fake_chat(system, user, **kw):
        seen["user"] = user
        return "Answer [1]."
    monkeypatch.setattr(gen_mod, "chat", fake_chat)
    state = {
        "question": "q?",
        "retrieved": [{"chunk_id": "c1", "document_name": "d.pdf", "page_number": 1,
                       "section_heading": "", "text": "evidence", "score": 1.0}],
        "trace": [],
    }
    gen_mod.generate(state)
    assert "PREVIOUS ATTEMPT" not in seen["user"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_generate.py -q`
Expected: new tests FAIL (no capping, no feedback block).

- [ ] **Step 3: Implement**

In `src/nodes/generate.py`:

1. Import `cap_pool` in both import blocks:

```python
    from src.nodes.search import cap_pool
```
(and `from nodes.search import cap_pool  # type: ignore` in the except branch)

2. In `generate`, cap the pool first — replace `retrieved = state.get("retrieved", [])` with:

```python
    retrieved = cap_pool(state.get("retrieved", []))
```

3. Build the user prompt with optional feedback — replace the `user = ...` line with:

```python
    user = f"Question: {state['question']}\n\nEvidence:\n{evidence}\n\n"
    feedback = state.get("verify_feedback", "")
    if feedback:
        user += (
            "PREVIOUS ATTEMPT FAILED VERIFICATION: "
            f"{feedback}. Rewrite the answer correcting this. Answer the "
            "question directly, use only the evidence, and cite with [n] "
            "markers that match the evidence numbers.\n\n"
        )
    user += "Answer with inline [n] citations:"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_generate.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nodes/generate.py tests/test_generate.py
git commit -m "feat: generate consumes capped pool and verify regeneration feedback"
```

---

### Task 6: Graph rewiring (verify node, failure-type edges, prepare_research)

**Files:**
- Modify: `src/graph.py`
- Test: `tests/test_graph.py` (update wiring tests that referenced critique/prepare_retry; follow the file's existing stub/monkeypatch pattern)

**Interfaces:**
- Consumes: `verify` (Task 4), `heal_action` state field, `retry_queries` consumption in search (Task 3).
- Produces: graph shape:

```
router: easy → direct_answer → END | meta → corpus_info → END
        medium → search → generate → verify
        hard   → decompose → search → generate → verify
verify (conditional on heal_action):
        none        → END
        regenerate  → generate            (same pool, feedback prompt)
        research    → prepare_research → search → generate → verify
```

- [ ] **Step 1: Update the graph code**

Replace `src/graph.py` content between the imports and `build_graph` (the two helper functions) and the wiring inside `build_graph`:

```python
def route_from_router(state: RAGState) -> str:
    route = state["route"]
    if route == config.ROUTE_EASY:
        return "direct_answer"
    if route == config.ROUTE_MEDIUM:
        return "search"
    if route == config.ROUTE_META:
        return "corpus_info"
    return "decompose"


def route_from_verify(state: RAGState) -> str:
    return state.get("heal_action", "none")


def prepare_research(state: RAGState) -> RAGState:
    """Turns verify's unsupported claims into sanitized retrieval queries.
    sub_questions is never touched: the coverage check keeps verifying the
    ORIGINAL decomposition, not the retry queries."""
    claims = state.get("unsupported_claims", [])
    queries = [re.sub(r"\[n?\d*\]", "", c).strip() for c in claims]
    queries = [q for q in queries if q]
    return {"retry_queries": queries or [state["question"]]}
```

Add `import re` at the top of graph.py. Change the critique import to `from src.nodes.verify import verify` (and the except-branch equivalent). Delete `route_from_critique` and `prepare_retry`.

New wiring inside `build_graph`:

```python
    graph.add_node("router", route_question)
    graph.add_node("direct_answer", direct_answer)
    graph.add_node("decompose", decompose)
    graph.add_node("search", search_node)
    graph.add_node("generate", generate)
    graph.add_node("verify", verify)
    graph.add_node("prepare_research", prepare_research)
    graph.add_node("corpus_info", corpus_info)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_from_router,
        {
            "direct_answer": "direct_answer",
            "search": "search",
            "decompose": "decompose",
            "corpus_info": "corpus_info",
        },
    )
    graph.add_edge("direct_answer", END)
    graph.add_edge("corpus_info", END)
    graph.add_edge("decompose", "search")
    graph.add_edge("search", "generate")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges(
        "verify",
        route_from_verify,
        {"none": END, "regenerate": "generate", "research": "prepare_research"},
    )
    graph.add_edge("prepare_research", "search")
```

Note the medium path now flows through verify too (`generate → verify` unconditionally); verify itself is cheap for medium (Stage 1 only, no LLM calls).

- [ ] **Step 2: Update tests/test_graph.py**

Read the existing file first. Update every reference: `critique` → `verify`, `prepare_retry` → `prepare_research`, `route_from_critique` → `route_from_verify`. The retry-loop test should now stub verify's output via `heal_action` instead of `critique_clean`/iterations, e.g.:

```python
def test_route_from_verify_dispatch():
    from src.graph import route_from_verify
    assert route_from_verify({"heal_action": "none"}) == "none"
    assert route_from_verify({"heal_action": "regenerate"}) == "regenerate"
    assert route_from_verify({"heal_action": "research"}) == "research"
    assert route_from_verify({}) == "none"


def test_prepare_research_sanitizes_markers():
    from src.graph import prepare_research
    state = {
        "question": "q?",
        "unsupported_claims": ["The Transformer uses Adam [3].", ""],
    }
    r = prepare_research(state)
    assert r["retry_queries"] == ["The Transformer uses Adam ."]  # marker stripped
```

Adapt end-to-end wiring tests in the file (which monkeypatch node functions) to the new node names, keeping their existing structure.

- [ ] **Step 3: Run the full offline suite**

Run: `pytest tests/ --ignore=tests/test_questions.py -q`
Expected: ALL PASS (including test_graph.py, and the Task-4 critique-import breakage is now resolved).

- [ ] **Step 4: Commit**

```bash
git add src/graph.py tests/test_graph.py
git commit -m "feat: rewire graph with verify node and failure-type-aware healing edges"
```

---

### Task 7: Surface verification warnings in the UI

**Files:**
- Modify: `app.py` (`_render_answer`)

**Interfaces:**
- Consumes: `result["verification_warnings"]` (list of strings).

- [ ] **Step 1: Implement**

In `app.py` `_render_answer`, after `st.write(result.get("answer", ""))`:

```python
    warnings = result.get("verification_warnings", [])
    if warnings:
        st.warning(
            "This answer did not pass all verification checks: "
            + "; ".join(warnings)
        )
```

And in the expander, change the hard-route block to also show the failure state:

```python
        if route == "hard":
            st.write(f"Iterations: {result.get('iterations', 0)}")
            st.write(f"Verified clean: {result.get('critique_clean')}")
```

- [ ] **Step 2: Verify no syntax errors**

Run: `python -c "import ast; ast.parse(open('app.py').read())"`
Expected: silent success.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: surface verification warnings in the chatbot UI"
```

---

### Task 8: Live verification against the acceptance criteria

**Files:**
- No source changes expected (tuning `COVERAGE_SIM_THRESHOLD` in `src/config.py` is allowed if criterion 4 fails).
- Modify: `BUILD_LOG.md` (append entry), `PROJECT_CHANGELOG.md` (new section)

This task validates the spec's §8 acceptance criteria against the LIVE system (Ollama must be running with qwen2.5:7b pulled).

- [ ] **Step 1: Offline suite green**

Run: `pytest tests/ --ignore=tests/test_questions.py -q`
Expected: ALL PASS.

- [ ] **Step 2: Routing harness does not regress**

Run: `python -m tests.test_questions`
Expected: routing accuracy ≥ 93.3% (28/30). Record the number.

- [ ] **Step 3: Re-run the 13-query suite**

Run the existing harness script (it invokes the same compiled graph the UI uses):

```bash
python /private/tmp/claude-501/-Users-apple-Desktop-Projects-aws-rag/1f2c836a-43e0-4695-85de-2004f41efd08/scratchpad/run_queries.py
```

If that scratchpad path no longer exists, recreate the 13 queries from the spec §1 evidence section as a local script under the scratchpad directory (same structure: invoke `build_graph()` per question, record route/citations/answer/iterations/warnings).

Check against acceptance criteria:
- Q1 (Transformer optimizer): answer names Adam / warmup / 4000, OR carries an honest could-not-verify note. Route stays hard.
- Q3 (PaLM) and Q6 (T5): route == medium, wall time ≈ 10-30s each.
- Q9 (ELECTRA): every number in the answer appears in its retrieved evidence (spot-check "1/4" vs "1/135").
- Q11/Q13 (3-way comparisons): all three documents cited or honestly flagged.
- No query errors or infinite loops; every hard query terminates within the budget.

- [ ] **Step 4: Validate answers against source PDFs**

For each factual claim in Q1, Q3, Q9, Q10 answers, grep the source PDF text (PyMuPDF) to confirm the value exists — the same validation methodology that caught the original Q9 fabrication.

- [ ] **Step 5: Tune the coverage threshold if needed**

If criterion 4 fails (coverage check misses shallow answers or force-loops good ones), sweep `COVERAGE_SIM_THRESHOLD` over {0.35, 0.40, 0.45, 0.50} against the Q11/Q13 answers and pick the value that separates covered from missed sub-questions. Update `src/config.py` with one line of justification in the comment.

- [ ] **Step 6: Record results**

Append a numbered entry to `BUILD_LOG.md` (append-only, never rewrite) summarizing: what was built (tasks 1-7), the live results per acceptance criterion with measured numbers, and any threshold tuning. Add a matching section to `PROJECT_CHANGELOG.md`.

- [ ] **Step 7: Final commit**

```bash
git add BUILD_LOG.md PROJECT_CHANGELOG.md src/config.py
git commit -m "Verify verification-gate acceptance criteria live; record results"
```
