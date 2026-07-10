# Design: Trustworthy Self-Healing — Verification Gate, Pool Control, and Routing Guard

**Date:** 2026-07-10
**Status:** Approved
**Scope:** Fix three validated failure classes: (1) the Mode-3 critique loop rubber-stamps
wrong or fabricated answers as clean, (2) multi-part answers are not comprehensively
covered, (3) the router over-escalates single-document questions from medium to hard.
**Explicit non-goal:** conversation/context memory (multi-turn follow-ups). Deliberately
excluded; nothing here forecloses it later.

---

## 1. Evidence (validated against source PDFs and live run data)

Thirteen medium/hard queries were run through the live graph; answers were validated
against the source PDF text, and the failing run was reproduced with full evidence
capture. Three failure classes, with root causes confirmed in code:

### F1 — Self-healing does not heal
- **Q1** ("What optimizer does the original Transformer use, and its warmup schedule?")
  escalated to hard, ran 3 iterations, reported `critique_clean: true`, and produced an
  answer about self-attention that never mentioned Adam, warmup, or 4000 steps (ground
  truth: attention paper p7). **Reproduced:** the evidence pool feeding generate held
  20 chunks across 8 documents; the correct Adam/warmup chunk (attention p7) WAS in the
  pool (2 of 20 slots; t5.pdf held 6). The evidence was present; synthesis drowned.
- **Root cause A (critique checks the wrong property):** `critique.py` verifies only
  "is every claim supported by the evidence?" The rambling answer's claims WERE
  supported by the polluted pool. Critique never asks "does the answer address the
  question?" — so a grounded non-answer passes.
- **Root cause B (pool poisoning):** `search_node` merges every iteration's hits into a
  growing pool; retries re-search and re-merge. Generation answers the original
  question against cross-corpus soup.
- **Root cause C (retry feeds statements as queries):** `prepare_retry` copies
  critique's `unsupported_claims` into `sub_questions` verbatim. Observed state:
  sub_questions containing "The original Transformer uses the Adam optimizer [3]." —
  declarative answers with citation markers, kept in state.
- **Q9** (ELECTRA sample efficiency) fabricated "1/135th of the pre-training compute";
  the paper says "less than 1/4". The number appears in no retrieved chunk. Critique
  passed it. **Root cause D:** no mechanism compares specific values in the answer
  against the evidence text; an LLM critique on the same 7B cannot be trusted to catch
  its own fabrication.

### F2 — Answers are not comprehensive
Three-way comparison questions (BERT/XLNet/ELECTRA; ALBERT/DistilBERT/Transformer)
returned answers deep on one document and a sentence on the others. **Root cause E:**
nothing ties the final answer back to the decompose sub-questions; synthesis can
silently drop a document and still pass critique.

### F3 — Routing over-escalates medium → hard
4 of 6 medium queries escalated to hard: ~10x latency (10.8s vs 80–112s), and in Q1 the
escalation caused the wrong answer (a scoped single-hop search would have surfaced the
p7 chunk dominantly). **Root cause F:** the router prompt defines hard as "multi-hop or
cross-document"; a two-part question about ONE document ("optimizer AND schedule")
pattern-matches "several searches" to a 7B. The crisp signal — how many distinct
indexed documents the question implicates — is rule-shaped and belongs in code.

**Design constraint (project lesson):** the local 7B (qwen2.5:7b) is unreliable at
fine-grained judgment. Crisp decisions move into deterministic code; the LLM is only
asked narrow binary questions. Escape hatch if the binary critique still proves
unreliable: a larger verifier model (qwen2.5:14b), named here, not assumed.

---

## 2. Architecture chosen

**Deterministic gate + narrow LLM critique** (chosen over claim-by-claim LLM
verification — 5–10 extra local calls per answer and dependence on 7B structured
output — and over a bigger verifier model, which fixes none of the structural causes).

Three cooperating changes:

1. A **routing guard** so single-document questions never enter the hard path.
2. **Evidence pool control** so generation sees focused, document-fair evidence.
3. A **verification gate** (deterministic stage + binary LLM stage) with
   **failure-type-aware healing** replacing the single critique verdict.

---

## 3. §1 Routing guard (deterministic, post-LLM-router)

New pure function in `src/ingest.py`:

- `implicated_documents(question) -> list[str]` — returns every indexed document
  whose FILENAME-STEM tokens overlap the question's tokens (reusing `_tokenize`
  including de-hyphenation, so "GPT-2" implicates gpt2.pdf). Stem-only, not title
  tokens: title-word matching is too noisy for implication ("the original
  Transformer" matches t5.pdf's title "…Text-to-Text Transformer" but not
  attention_is_all_you_need — scoping to the wrong paper is worse than not scoping).

Rule in `router.py`, applied after the LLM label (mirroring the existing easy→medium
upgrade):

- LLM says `hard` AND `len(implicated_documents(q)) == 1` → **downgrade to `medium`**.
- ≥2 implicated documents → stays hard. 0 implicated documents → stays hard
  (corpus-wide synthesis, or a document referenced by unmatchable paraphrase — a
  safe, slower-not-wrong failure).
- No medium→hard upgrade rule: under-escalation was not observed (YAGNI).

Validated against the run: PaLM-count and T5-text-to-text (1 implicated doc each)
route medium; RoBERTa-vs-BERT names 2 docs and stays hard (acceptable: comparative);
Transformer-optimizer says "Transformer" (a paraphrase matching no stem) → implicates
0 docs and stays hard — it is healed by the verification gate instead (§5–6); all
genuinely-hard queries (≥2 named docs or 0 named docs) stay hard.

Trace line: `router → hard downgraded to medium (implicates only <doc>)`.

## 4. §2 Evidence pool control

In `search.py` / config:

- `POOL_CAP = 12` — the pooled evidence list passed to generate is capped at 12 chunks.
- `PER_DOC_CAP = 4` — fill the pool in score order, skipping any chunk whose document
  already holds 4 slots; if the pool is still under `POOL_CAP` after that pass,
  backfill with the highest-scored skipped chunks. The per-doc quota is what protects
  N-way comparisons (F2): one paper cannot evict another's evidence entirely.
- Page-1 boost chunks compete under the same caps (they carry scores).
- **Regeneration retries do not re-search and do not grow the pool** (see §6).
  Re-search retries merge as today, then re-cap.

## 5. §3 Verification gate

New node `verify` replacing critique's monopoly. Two stages.

### Stage 1 — deterministic, zero LLM calls, always runs

1. **Citation check:** every `[n]` marker in the answer resolves to a real evidence
   index (1..len(pool)); at least one citation present. Malformed markers (`[n14]`,
   bare `[n]`) fail. → `failure_type: "citations"`.
2. **Number-grounding check:** every numeric value in the answer must appear in the
   retrieved evidence text, under normalization:
   - thousands separators stripped ("4,000" ≡ "4000")
   - magnitude words/suffixes folded ("175B" ≡ "175 billion" ≡ "175-billion")
   - percentages and fractions matched as written ("93.3%", "1/4", "1/135")
   - page/citation-marker digits and the evidence-index digits themselves excluded.
   A number in the answer with no normalized match anywhere in the pool text →
   `failure_type: "fabrication"` with the offending values. (Catches Q9's "1/135".)
3. **Coverage check (hard route only):** for each ORIGINAL decompose sub-question,
   compute max cosine similarity between the sub-question embedding and the answer's
   sentence embeddings, using the already-loaded MiniLM model. Any sub-question below
   `COVERAGE_SIM_THRESHOLD` (start 0.45, tuned empirically against the real corpus) →
   `failure_type: "coverage"` with the missed sub-questions. (Catches F2; costs
   milliseconds, no LLM.)

### Stage 2 — LLM, binary-only, runs only if Stage 1 passes

4. **Responsiveness:** one call, `max_tokens≈4`:
   "Question: … Answer: … Does the answer directly address what the question asked?
   yes/no". `no` → `failure_type: "responsiveness"`. (Catches Q1's grounded
   non-answer — the property the old critique never checked.)
5. **Support:** the existing critique prompt (claims vs. evidence) over the capped
   pool. Unsupported → `failure_type: "support"` with restated claims.

`clean` requires **all five** to pass. The gate emits `failure_type`,
`verify_feedback` (human-readable specifics for the regeneration prompt), and for
support-failures the restated claims.

## 6. §4 Failure-type-aware healing

The conditional edge after `verify` routes on failure type, replacing the single
retry path:

| failure_type | Action |
|---|---|
| `citations`, `fabrication`, `responsiveness`, `coverage` | **Regenerate:** call generate again with the SAME capped pool plus explicit feedback appended to the prompt ("Your previous answer failed verification: it contained the unsupported value 1/135 / it did not address: <sub-questions>. Correct this; use only the evidence."). No search. Pool unchanged. |
| `support` | **Re-search:** restated claims go into a NEW state field `retry_queries` — sanitized in code (citation markers stripped) — and search runs on those. `sub_questions` is never overwritten (fixes root cause C: coverage keeps checking the original decomposition). Pool merges then re-caps. |
| clean | END |
| budget exhausted | **Honest exit** (below) |

**Budget:** `iterations` counts retrieval passes plus regenerations: `search_node`
keeps incrementing it exactly as today (covers initial search and re-searches), and
`verify` additionally increments it when dispatching a regeneration (which bypasses
search). The router still resets it to 0 per run. Healing is allowed while
`iterations < budget`, where budget is `MAX_ITERATIONS = 3` for hard (identical cap
semantics to today) and `2` for medium (initial search + one regenerate). Every heal
cycle increments the counter by at least one, guaranteeing termination.

**Honest exit:** on exhaustion, if the latest failure is `support` or `coverage` with
no new evidence found by the last re-search, the answer is replaced with an explicit
"the indexed documents do not contain enough to answer X" statement for the unmet
parts, keeping the supported parts. Otherwise the best answer is returned with
`verification_warnings` populated. `app.py` renders warnings as a caution caption
("This answer did not pass all verification checks: …") instead of presenting the
answer as clean.

**Medium path:** medium answers get Stage-1 checks 1–2 (no sub-questions → no
coverage; no Stage 2 to preserve latency) + at most ONE regenerate retry, then the
same honest-exit rule. Zero added latency when clean (pure code); ~10s only when an
answer is actually broken. Fabrication does not respect routing labels.

## 7. §5 State and config additions

`RAGState` gains: `failure_type: str`, `verify_feedback: str`,
`retry_queries: list[str]`, `verification_warnings: list[str]`.
Config gains: `POOL_CAP = 12`, `PER_DOC_CAP = 4`, `COVERAGE_SIM_THRESHOLD = 0.45`.
Graph: `critique` node is replaced by `verify`; `prepare_retry` is replaced by the
failure-type conditional + a `prepare_research` node that fills `retry_queries`.

## 8. Acceptance criteria

1. **Q1** ends with an Adam / 4000-warmup / inverse-sqrt answer or an honest
   "not found" — never a grounded non-answer marked clean. (It stays routed hard —
   "the original Transformer" matches no filename stem — so the verification gate,
   not the routing guard, must heal it: responsiveness check → regenerate.)
2. **Q9** emits no numeric value absent from its retrieved evidence.
3. The 2 stem-matchable mis-escalated medium queries (PaLM count, T5 text-to-text)
   route medium (~10s each, not 80–112s); RoBERTa-vs-BERT and Transformer-optimizer
   stay hard by design (2 named docs / paraphrase); all genuinely-hard queries stay
   hard.
4. N-way comparison answers cover every named document (coverage check passes or the
   answer honestly states what is missing).
5. Full 13-query suite re-run and validated against source PDF text; routing accuracy
   on the 30-question harness does not regress below 93.3%.
6. Offline unit tests (no LLM): number normalization/matching, citation resolution,
   coverage similarity, `implicated_documents`, routing-guard downgrade,
   retry-query sanitization, pool capping with per-doc quota, honest-exit wording.

## 9. Residual risks (accepted, named)

- The number-grounding check does not catch fabricated **entity names** (only values).
- `COVERAGE_SIM_THRESHOLD` needs empirical tuning; a bad threshold either misses
  shallow answers (too low) or forces needless regeneration (too high). Tune against
  the real corpus before locking.
- The binary responsiveness check still trusts the 7B on one narrow judgment. If it
  proves unreliable in live verification, the named escape hatch is a larger local
  verifier model (qwen2.5:14b) for that single call.
- The routing guard depends on `implicated_documents` recall: a question referencing a
  document by unmatchable paraphrase counts as 0 docs and stays hard — a safe
  (slower, not wrong) failure.
