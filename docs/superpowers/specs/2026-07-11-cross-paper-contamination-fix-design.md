# Cross-paper retrieval contamination fix

## Context

A production-readiness audit (run after expanding the corpus from 2 to 10 papers,
1490 chunks) identified that `src/nodes/search.py`'s FAISS search has zero
document-level filtering — it's pure semantic similarity across the entire corpus.
With only 2 papers this rarely mattered; with 10 vocabulary-similar transformer
papers (BERT, RoBERTa, ALBERT, ELECTRA, DistilBERT are close siblings), a
sub-question clearly about one paper (e.g. "How does BERT handle positional
encoding?") can now surface top-k chunks from a sibling paper ranked higher than
the actually-relevant paper's chunks. This feeds noisy, off-topic evidence into
`generate` and `critique`, degrading answer quality and potentially causing the
Mode 3 critique loop to burn iterations on evidence that was never going to satisfy
the claim.

This is one of five prioritized fixes from that audit (retry/timeout hardening was
fix #1, already shipped). This is fix #2.

## Decision: detect a named paper in the query text, filter to that document; fall back to whole-corpus search when no confident match exists

Mode 3's `decompose` step deliberately generates genuinely cross-paper sub-questions
for real multi-hop comparisons (e.g. one sub-question per paper being compared) —
filtering must not break that case. The fix must only kick in when a sub-question
(or Mode 2's single question) is clearly about one specific paper, and must stay out
of the way otherwise. An unconfident or ambiguous match must default to searching
the whole corpus (today's behavior), never to guessing wrong and filtering out the
correct paper.

## Changes

### `src/ingest.py`

1. **`Chunk` dataclass gains `document_title: str`** — extracted once per PDF
   (identical value across every chunk of that document), not derived per-chunk.
2. **New `_extract_title(pdf_path: Path) -> str`.** Opens page 1 with
   `page.get_text("dict")`, finds the text span with the largest font size on that
   page (the standard, reliable heuristic for an academic paper's title — this is
   the same "font-size analysis" the existing heading-detection code already
   identified as "more precise but heavier," and it's worth the one-time per-document
   cost here since it runs once per PDF, not once per chunk). Falls back to the
   filename stem (e.g. `"bert"` from `bert.pdf`) if no usable text span is found
   (e.g. a scanned/image-only first page).
3. **`search()` gains an optional `document_hint: str | None = None` parameter.**
   When `None` (the default), behavior is completely unchanged — searches the whole
   corpus, exactly as today. When provided:
   - Over-fetch from FAISS (request more than `k` candidates, e.g. `k * 4`) to leave
     enough headroom after filtering.
   - Filter results to only `document_name == document_hint`.
   - If fewer than `k` results remain after filtering (the matched document doesn't
     have enough relevant chunks for this query), that's fine — return what's
     available rather than re-querying further; a document-scoped search returning
     fewer than `k` results is expected behavior, not an error.
4. **New `match_document(text: str) -> str | None`.** Tokenizes `text` and, for
   every document currently in the index, tokenizes `document_title + " " +
   filename_stem` — lowercase word splitting on both sides. Scores by token overlap.
   Returns the best-matching `document_name` **only if it's an unambiguous winner**
   (meaningfully higher overlap than every other document — the exact margin is an
   implementation-time tuning choice, to be validated against real queries during
   testing, not hardcoded speculatively here). Returns `None` when no document
   stands out clearly, which is the safe default and preserves today's whole-corpus
   search behavior.

### `src/nodes/search.py`

`search_node` already builds a `queries` list (`sub_questions` if present, otherwise
`[state["question"]]`) and loops over it calling `faiss_search(q, k=config.TOP_K)`.
This loop changes to compute `document_hint = match_document(q)` for each query and
pass it through: `faiss_search(q, k=config.TOP_K, document_hint=document_hint)`.

This applies uniformly to **both Mode 2 (single-hop) and Mode 3 (multi-hop)**, since
both go through this same loop today — no branching needed to distinguish them, and
Mode 2 benefits from the same scoping (e.g. "What pretraining tasks does BERT use"
correctly narrows to `bert.pdf` even in a single-hop search).

### No changes to `decompose.py`, `generate.py`, `critique.py`, `graph.py`, `app.py`

`decompose` keeps generating whatever sub-questions it already generates — the new
matching step in `search_node` decides whether to scope each one, without
`decompose` needing to know or care about document filtering.

## Re-ingestion required

The current index (`index/faiss.index` + `index/metadata.json`, built from the
10-paper corpus) has no `document_title` field on any existing chunk. This fix
requires a full re-ingestion (`python -m src.ingest`) to backfill titles before
`match_document()` has anything to match against. Per discussion, this is a
**manual step after implementation**, not automated as part of the code change —
consistent with how every other ingestion run in this project has been an explicit,
visible step.

## Testing approach

1. **Offline unit tests** for the pure logic, no Groq/FAISS needed at the unit
   level where possible:
   - `_extract_title`: test against a hand-built or existing sample PDF, confirm it
     returns a sensible title, not empty/garbage.
   - `match_document`: test with a hand-built small set of fake documents (titles +
     filenames) and confirm: a clearly-named paper matches correctly; an ambiguous
     or unrelated query returns `None`; a query naming no paper at all returns `None`.
2. **Live verification after re-ingestion:**
   - Run `python -m src.ingest` to rebuild the index with titles for the real
     10-paper corpus.
   - Spot-check `match_document()` against real sub-question-shaped text for at
     least 3 of the 10 papers, confirming correct matches.
   - Run a real Mode 2 query clearly about one paper (e.g. the existing BERT
     pretraining-tasks preset question) and confirm citations now come exclusively
     from that paper's `document_name`, whereas before this fix cross-paper
     contamination was a possibility (not necessarily guaranteed to have occurred on
     that exact question — the point is to confirm the filter is active and correct
     now, not to prove the bug was reproducible before the fix).
   - Run a real Mode 3 query that's a genuine multi-hop comparison across two named
     papers (e.g. the existing BERT-vs-Transformer preset question) and confirm each
     sub-question still gets correctly scoped to its own paper, and the final answer
     still cites both papers (proving the fix does not break legitimate cross-paper
     comparison).

## Out of scope

- Heading-detection quality (86% of chunks mislabeled "Abstract"/"References"/
  "Acknowledgments") — a separate, already-identified audit finding, not addressed
  by this fix (title extraction is a related but distinct mechanism from heading
  detection and does not fix it).
- Router/decompose prompt-corpus mismatch (still referencing papers not in the
  corpus) — separate audit finding, unaddressed here.
- Phase 7's 30-question test set — separate, unaddressed here.
- Any change to how `IndexFlatIP` performs search itself (still a flat, exact index
  — confirmed fast enough at this scale by the audit; this fix only adds
  post-filtering on top of existing search results, not a different index type).
