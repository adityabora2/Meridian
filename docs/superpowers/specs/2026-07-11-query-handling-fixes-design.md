# Query-handling fixes: corpus meta-questions, metadata retrieval, document-aware routing

## Context

Live testing of the chatbot UI surfaced three real query-handling gaps, each with a
distinct root cause:

1. **"what documents are ingested?"** routed to retrieval, found nothing (correctly —
   no document's *content* lists the corpus), and gave up. There is no capability to
   answer questions about the corpus/system itself.
2. **"in xlnet who are the authors"** returned no useful answer even though the author
   names ARE indexed. Confirmed: the first page-1 chunk of `xlnet.pdf` contains
   "Zhilin Yang, Zihang Dai, ..." verbatim, but pure semantic search ranks a raw
   list of names far below the query "who are the authors", so that chunk never
   reaches the top-k. This is a known weak spot of semantic RAG on metadata queries.
3. **"explain bert"** answered from the model's own general knowledge (Mode 1, no
   retrieval) instead of the BERT document in the corpus. The routing was a defensible
   LLM judgment, but for a document Q&A system, a question naming an indexed document
   should be grounded in that document, not the model's memory.

All three are fixed with precise, reuse-existing-infrastructure changes rather than
broad prompt loosening (which would hurt the currently-strong 93.3% routing accuracy).

## Fix A — Corpus meta-questions (new `meta` route + `corpus_info` node)

- Add a fourth router category `meta`, for questions about the corpus or the system
  itself (not about document content): "what documents are loaded?", "what files can
  I ask about?", "which papers do you have?".
- `src/config.py`: add `ROUTE_META = "meta"` and a `MODE_LABELS[ROUTE_META]` entry
  (e.g. "Corpus Info"). The router's `_parse_label` and its fallback logic must
  recognize the new label.
- `src/nodes/router.py`: extend `_SYSTEM` with the `meta` category and examples;
  extend `_parse_label` to accept `meta`. The existing fail-safe fallback to `medium`
  is unchanged (an unparseable label still routes to retrieval, never to `meta` by
  accident).
- New file `src/nodes/corpus_info.py` with `corpus_info(state) -> RAGState`: reads the
  index metadata via `ingest.load_index()`, collects the distinct
  `(document_name, document_title)` pairs, and returns a formatted `answer` listing
  them (plus a count). No LLM call, no retrieval — deterministic. `citations` is `[]`.
  On an empty/missing index it returns a clear "no documents indexed yet" message
  rather than raising.
- `src/graph.py`: register the `corpus_info` node; the router conditional edge gains a
  `meta -> corpus_info` branch; `corpus_info -> END`.

## Fix B — First-page boosting for document-scoped queries

- Root cause: metadata (authors, title, affiliations, abstract) lives on page 1, but
  those chunks rank low semantically for metadata-shaped queries.
- Change: when a query is confidently scoped to one document (i.e.
  `match_document(query)` returns a document name), the search for that query must
  include that document's **page-1 chunk(s)** in the candidate pool, merged with the
  normal semantic top-k (deduplicated by `chunk_id`, as the existing `_merge` already
  does in `search_node`). This guarantees the title/author/abstract block is available
  to `generate` for metadata questions about a named document.
- Scope/bounding: this only triggers when there is a confident document match, so it
  never adds page-1 noise to general (unscoped) queries. Implementation lives in
  `src/ingest.search()` (which already knows `document_hint`) or `search_node` —
  decided at implementation time based on which keeps the boundary cleanest; the
  page-1 chunks are fetched from the loaded metadata, not via a second FAISS call.
- The number of page-1 chunks added is bounded (page 1 typically yields only a few
  chunks) so the evidence set stays a reasonable size for the LLM prompt.

## Fix C — Route to retrieval when a query names an indexed document

- After the LLM router returns its label, if the label is `easy` **and**
  `match_document(question)` returns a confident document match, upgrade the route to
  `medium`. Rationale: a question naming a document in the corpus should be answered
  from that document (with citations), not from the model's parametric memory.
- Genuine general-knowledge questions (which name no indexed document — e.g. "what is
  machine learning") match nothing, stay `easy`, and answer directly. So the strong
  Mode-1 behavior on the routing test set is preserved.
- Implementation: a small post-processing step in `route_question` (after parsing the
  label, before returning), reusing the existing `match_document`. The upgrade is
  logged in the trace (e.g. "router → easy upgraded to medium (names <doc>)") so the
  behavior is visible and debuggable.
- Interaction with Fix A: the `meta` route is decided by the LLM router and is not
  subject to this `easy→medium` upgrade (a meta-question about the corpus should not
  be forced into content retrieval). Only an `easy` label is eligible for upgrade.

## No changes to

`decompose.py`, `critique.py`, `direct_answer.py`, `generate.py`, `app.py`, or the
Mode 3 critique loop. The three fixes are additive to routing, a new node, and the
search candidate set.

## Testing approach

1. **Offline unit tests:**
   - `corpus_info`: with a monkeypatched/loaded small metadata set, returns the
     expected document list; handles the empty-index case gracefully.
   - Router `meta` parsing: `_parse_label("meta")` resolves; an unparseable label
     still falls back to `medium`, never `meta`.
   - Fix C upgrade logic: `easy` + a confident `match_document` → `medium`; `easy` +
     no match → stays `easy`; a `meta` label is never upgraded. (Via monkeypatched
     `chat` and `match_document`.)
   - Fix B: a document-scoped search includes the matched document's page-1 chunk in
     its results even when that chunk's semantic score alone would exclude it (via a
     small hand-built index or monkeypatched metadata).
2. **Live verification** — re-run all four originally-failing queries and confirm:
   - "what documents are ingested?" lists the 11 documents.
   - "in xlnet who are the authors" now surfaces the author-block chunk and answers
     with the actual author names.
   - "explain bert" routes to retrieval and answers from `bert.pdf` with a citation.
   - A normal content question and a genuine general-knowledge question still behave
     correctly (no regression).
3. **Routing-accuracy regression:** re-run `tests/test_questions.py` and confirm the
   easy/medium/hard accuracy does not degrade materially (the 2 medium "misses" that
   name documents may now legitimately shift, so the harness's expected labels for
   any document-naming easy/medium questions should be reviewed for consistency with
   the new Fix C behavior).

## Out of scope

- Cross-question conversational memory (each question is still routed and answered
  independently; the chat history is visual only).
- Improving semantic retrieval quality in general beyond the targeted page-1 boost.
- Any change to the embedding model, chunking, or FAISS index type.
