# Generalize from research papers to general PDF documents

## Context

The project was originally scoped (per the initial PLAN.md spec) around "AI research
papers" specifically — the ingestion heading-detection heuristic, the router and
decompose LLM prompts, and even the `data/papers/`/`PAPERS_DIR` naming all assume the
corpus is academic papers. The user pushed back on this: PyMuPDF's parsing and
layout extraction is genuinely capable of handling any PDF document type, so hard-
coding academic-paper assumptions throughout the pipeline is an unnecessary and
misleading scope restriction, not a technical limitation.

This work also absorbs a previously-paused, already-designed fix: cross-paper
retrieval contamination (a production audit finding — with 10 vocabulary-similar
papers, FAISS search with no document-level filtering can surface a sibling
document's chunks ranked above the actually-relevant one). That fix's design
depended on per-document title extraction, which this generalization effort
redesigns from scratch anyway — so the two are combined into one spec rather than
building title extraction twice.

Verified live before designing: PyMuPDF's embedded metadata (`doc.metadata['title']`,
`['author']`, etc.) is **empty on all 10 current papers** — they're all LaTeX-
generated PDFs that strip this metadata. This is common for academic PDFs
specifically; other real-world PDFs (Word/PowerPoint exports, many reports) often do
populate this metadata. The design below handles both cases.

## Change 1 — Font-size-based heading detection (`src/ingest.py`)

**Replaces** the current keyword-regex approach entirely
(`_HEADING_KEYWORDS`/`_HEADING_NUMBERED` in `parse_pdf`'s current implementation),
which only recognizes academic section names (`abstract`, `introduction`,
`methodology`, `references`, etc.) and is already measured at 86% wrong even on the
10 research papers it was designed for.

- New `_extract_font_sizes(page) -> list[tuple[float, str]]`: uses
  `page.get_text("dict")` to get each text span's font size paired with its text.
- New body-text baseline detection: sample font sizes across the first several pages
  of the document to find the dominant/most common size — this is the document's
  "normal text" size.
- **Heading rule:** a text block is treated as a heading if its font size is
  meaningfully larger than the detected body-text baseline (exact ratio/threshold is
  an implementation-time tuning decision, validated empirically against both the
  existing 10 papers and at least one non-paper test PDF during implementation, not
  fixed speculatively here).
- This is domain-agnostic by construction: a heading is defined by how it looks
  (visually larger/more prominent than surrounding body text), not by what words it
  contains — equally correct for "3.2 Results," "Chapter 4: Troubleshooting," or
  "SECTION 7 — TERMINATION."
- The old keyword-based heuristic is deleted, not kept as a parallel/fallback path —
  font-size detection is expected to strictly improve on it, including for the
  academic-paper case, so maintaining two heading-detection code paths would add
  complexity without benefit.

## Change 2 — Per-document title extraction (`src/ingest.py`)

- `Chunk` dataclass gains `document_title: str` — one value computed once per
  document and repeated across every chunk of that document (not derived per-chunk).
- New `_extract_title(doc: fitz.Document, pdf_path: Path) -> str`:
  1. Check `doc.metadata["title"]` — if non-empty and not obviously a placeholder
     (e.g. not empty string, not equal to the filename), use it directly. This is
     the correct, cheap answer for PDFs that have real embedded metadata.
  2. Otherwise, fall back to the largest-font-size text block on page 1, reusing the
     font-size infrastructure from Change 1 — the standard, reliable heuristic for a
     document's title when embedded metadata is absent (true for all 10 current
     LaTeX-generated papers).
  3. Final fallback: the filename stem (e.g. `"bert"` from `bert.pdf`), if neither
     of the above yields usable text (e.g. a scanned/image-only first page with no
     extractable text spans at all).

## Change 3 — Cross-paper (cross-document) retrieval contamination fix (`src/ingest.py`, `src/nodes/search.py`)

Folded in from the previously-paused design, now built on the general-document title
mechanism above instead of a paper-specific one.

- `search()` in `src/ingest.py` gains an optional `document_hint: str | None = None`
  parameter. Default behavior (whole-corpus search) is unchanged. When provided:
  over-fetch more candidates than `k` from FAISS, filter to only
  `document_name == document_hint`, return what's available (fewer than `k` results
  after filtering is expected and fine, not an error).
- New `match_document(text: str) -> str | None` in `src/ingest.py`: tokenizes the
  input text and, for every currently-indexed document, tokenizes
  `document_title + " " + filename_stem`. Scores by word-token overlap. Returns the
  best-matching `document_name` only if it's an unambiguous winner (clearly higher
  overlap than every other document); returns `None` otherwise — the safe default
  that preserves today's whole-corpus search behavior for ambiguous or genuinely
  cross-document queries.
- `search_node` in `src/nodes/search.py`: for each query in its existing `queries`
  list (used identically by both Mode 2's single question and Mode 3's
  decompose-generated sub-questions), compute `document_hint = match_document(q)`
  and pass it through to `faiss_search`. No branching needed between Mode 2 and Mode
  3 — both already share this same loop.
- No changes to `decompose.py`, `generate.py`, `critique.py`, or `graph.py` — the
  matching/filtering decision lives entirely in `search_node` and `ingest.py`.

## Change 4 — Domain-agnostic prompts (`src/nodes/router.py`, `src/nodes/decompose.py`)

Both system prompts currently frame the system as being about "AI research papers
(Adaptive-RAG, Self-RAG, Chain-of-Verification, and related work)" with paper-
specific example questions. Rewrite both to:
- Describe the system generically: "a document Q&A system over an indexed document
  collection" (or equivalent neutral phrasing), with no reference to research papers
  or any specific topic.
- Replace example questions with **shape-based**, domain-neutral examples that
  illustrate the easy/medium/hard distinction by structure, not content — e.g. a
  generic definitional question for "easy," a single-document-lookup-shaped question
  for "medium," and a cross-document-comparison-shaped question for "hard" — without
  naming any specific real document or topic.
- No runtime coupling to the actual indexed documents — prompts remain static text,
  not dynamically constructed from the current corpus. This keeps routing/decompose
  behavior predictable and avoids a runtime dependency on the index being loadable
  just to build a prompt string.

## Change 5 — Rename `papers` terminology to `documents`

Since the system now genuinely supports general documents, keeping "papers" naming
would be misleading to future readers:
- `data/papers/` → `data/documents/` (directory rename).
- `config.PAPERS_DIR` → `config.DOCS_DIR` in `src/config.py`.
- Update user-facing strings in `ingest.py` (CLI output, the "drop PDFs and re-run"
  error message, etc.) from "papers" to "documents" where the meaning is general.
- `PLAN.md` and `BUILD_LOG.md` are append-only historical logs, per this project's
  established convention — past entries are **not** rewritten to retroactively say
  "documents" instead of "papers" (that would misrepresent what was actually built
  and decided at the time). This change gets its own new BUILD_LOG entry describing
  the terminology shift going forward, consistent with how every other decision in
  this project has been logged.
- The existing 10 PDFs (all research papers) are still valid content for
  `data/documents/` — moving the directory doesn't require different files, just a
  new location.

## Testing approach

1. **Offline unit tests:**
   - Font-size heading detection: test against the existing 10 real papers (as a
     regression/improvement check against the previously-measured 86% mislabeling
     rate) and, if feasible, a synthetic multi-section PDF with deliberately varied
     font sizes.
   - Title extraction: test the metadata-present case (a PDF with a real embedded
     title, if available/constructible) and the metadata-absent case (the current
     10 papers, all of which have empty embedded titles — confirmed live).
   - `match_document`: hand-built test documents/titles, confirming a clear match
     resolves correctly, an ambiguous/unrelated query returns `None`.
2. **Live verification after re-ingestion:**
   - Re-run ingestion (`python -m src.ingest`) against `data/documents/` (post-
     rename) with the current 10 papers, confirm the index builds successfully with
     the new schema (`document_title` present on every chunk).
   - Measure the new heading-detection accuracy against the same corpus and compare
     to the previously-measured 86% mislabeling rate — this is the direct,
     quantified proof that font-size detection is a real improvement, not just a
     different heuristic.
   - Spot-check `match_document()` and a document-scoped search against several of
     the 10 papers.
   - Run the existing Mode 2 and Mode 3 preset questions end-to-end and confirm
     citations, routing, and answers are still correct after all these changes.
   - **Generalization proof:** ingest at least one genuinely non-research-paper PDF
     (the user should supply one, or a stand-in like a public manual/report/spec
     document) alongside or instead of the existing papers, and confirm heading
     detection, title extraction, and Q&A all work correctly on it — this is the
     concrete evidence that the system no longer assumes academic-paper structure.

## Change 6 — Fix a stray "papers" reference in `generate.py`

`generate.py`'s no-evidence fallback answer currently reads "I couldn't find
relevant evidence in the indexed papers to answer this." — change "papers" to
"documents" for consistency with Change 5. This is the only "papers"/research-
specific string found in `generate.py` or `critique.py` (verified by direct grep of
both files); no other changes needed to either file — both operate purely on
already-retrieved chunk dicts and generic prompts.

## Out of scope

- Any other change to `generate.py`, `critique.py`, or `graph.py` beyond Change 6's
  one-string fix.
- Phase 7's 30-question test set (still pending, separately).
- A full production-grade document-type classifier or per-type parsing strategies
  (e.g. detecting "this is a legal contract" and applying contract-specific logic) —
  out of scope; the goal is removing academic-paper-specific assumptions, not adding
  new per-type specialization.
