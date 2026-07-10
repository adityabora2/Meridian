# Structured console logging

## Context

The system currently has **no real logging**. Visibility into what happens on a query is
limited to: (1) the `trace` list each node appends to, shown in the Streamlit UI's "How
this answer was produced" expander *after* a query finishes, and (2) `print()` statements
that only fire during `python -m src.ingest` (ingestion) and in nodes' `__main__` test
blocks. During a live app query there is no terminal output, no timings, no persistent
record, and no way to watch a query execute step by step.

The goal: **structured console logging you watch live in the terminal** as a query runs —
which node is firing, the route decision, retrieval hits and scores, and (critically) how
long each LLM call takes, since the local model is the slow part and Mode 3 makes up to 8
calls per question.

## Decisions (from brainstorming)

- **Concise by default (INFO), verbose on opt-in (DEBUG).** One readable line per node at
  INFO; full detail (retrieved chunk texts + scores, sub-question texts, raw router
  reply) only at DEBUG.
- **Level controlled by a `LOG_LEVEL` env var**, defaulting to `INFO`. E.g.
  `LOG_LEVEL=DEBUG streamlit run app.py`. No code edit needed to change verbosity; works
  for both the app and the `python -m src.ingest` CLI.
- **Additive to the existing `trace`, not a replacement.** The `trace` list stays exactly
  as-is (it powers the UI expander and is part of `RAGState`). Logging is an independent
  second channel to the terminal. Every node keeps its `trace.append` AND gains logging
  calls. This avoids any change to the verified graph state shape.

## Components

### `src/logging_config.py` (new)

- `setup_logging() -> None`: reads `os.getenv("LOG_LEVEL", "INFO")`, configures the
  `adaptive_rag` logger namespace with a `StreamHandler` to stderr and a clean formatter
  (`%(asctime)s %(levelname)s %(name)s: %(message)s`, time as `HH:MM:SS`). **Idempotent**
  — guarded so repeated calls (Streamlit reruns the whole script on every interaction) do
  not attach duplicate handlers. Does not touch the root logger's third-party handlers
  beyond what's needed (so the existing harmless huggingface warnings aren't amplified).
- `get_logger(name: str) -> logging.Logger`: returns `logging.getLogger(f"adaptive_rag.{name}")`.
- Invalid `LOG_LEVEL` values fall back to `INFO` (do not crash on a typo'd env var).

### Call sites for `setup_logging()`

- Top of `app.py` (once, before `build_graph()`).
- `src/ingest.py`'s `main()` (so CLI ingestion also gets structured logs, though its
  existing `print()`s stay — they're the user-facing progress output).

### Per-node logging (all additive; every existing `trace.append` stays)

Each node module gets a module-level `log = get_logger("<node>")`. Logged at **INFO**:

- **router.py**: the question (truncated), whether meta-detection fired, the route label,
  and any `easy → medium` upgrade with the matched document. DEBUG: the raw LLM reply.
- **search.py**: per query — whether it was document-scoped (and to which doc), hit
  count, top score, page-1-boost count, pooled total, iteration number. DEBUG: each
  retrieved chunk's `document_name p<page>` + score.
- **generate.py**: evidence-chunk count in, citation count out. DEBUG: the resolved
  citations.
- **decompose.py**: number of sub-questions. DEBUG: the sub-question texts.
- **critique.py**: the verdict (clean / unsupported) and unsupported-claim count. DEBUG:
  the unsupported claim texts.
- **corpus_info.py**: document count (or the no-index case).

### LLM call timing (`src/nodes/llm.py`)

- `chat()` wraps the `client.chat(...)` call with a monotonic timer and logs at INFO:
  the elapsed seconds and a short label for which call it was. Since `chat()` has no
  inherent knowledge of *which* node called it, add an optional `label: str | None = None`
  keyword parameter (default `None`) so callers can pass a short tag (e.g.
  `chat(..., label="router")`); the log reads e.g. `chat(router) took 0.4s`. When no label
  is passed it logs `chat() took 0.4s`. This keeps the timing at the single choke point
  every LLM call already flows through, rather than scattering timers across nodes.
  - Callers updated to pass a label: router, decompose, generate, critique
    (direct_answer too). This is a small, backward-compatible signature addition
    (`label` defaults to `None`, so nothing breaks if omitted).

## Example INFO output for one Mode-2 query

```
14:22:01 INFO adaptive_rag.router: Q='who are the authors in xlnet' route=medium
14:22:01 INFO adaptive_rag.search: query scoped->xlnet.pdf | 6 hits, top=0.62, +5 page-1, 11 pooled (iter 1)
14:22:04 INFO adaptive_rag.llm: chat(generate) took 3.1s
14:22:04 INFO adaptive_rag.generate: 11 evidence chunks -> 3 citations
```

At `LOG_LEVEL=DEBUG`, the same query additionally logs the retrieved chunks with scores,
and any node that has DEBUG detail emits it.

## Constraints

- **No new dependencies** — Python's `logging` is stdlib.
- **No emoji, no em/en dashes** in any log message string (same project constraint as
  user-visible text; log lines are effectively user-facing in the terminal). Use `->` not
  arrows, `-` not em-dashes.
- **No change to `RAGState`, the `trace` mechanism, the graph topology, or any node's
  return value.** Logging is side-effect-only.
- **No behavior change** — if logging were removed, the system would produce identical
  answers. Logging never alters control flow.
- Log messages must not dump full document text at INFO (only counts/scores/timings);
  full text is DEBUG-only, so a normal run stays readable.

## Testing

1. **Offline unit tests** (`tests/test_logging_config.py`):
   - `setup_logging()` respects `LOG_LEVEL` (INFO default; DEBUG when set; invalid value
     falls back to INFO).
   - Idempotent: calling it twice does not attach duplicate handlers (assert handler
     count stable).
   - A representative node emits a log record without changing its return value — e.g.
     capture records via a handler/`caplog` while calling `route_question` (with
     monkeypatched `chat`) and assert both that the route is correct AND a log record was
     emitted. This proves logging is additive, not disruptive.
2. **Live verification:**
   - Run a real query through the app (or a `build_graph().invoke` script) and confirm the
     terminal shows the expected concise INFO lines including an LLM timing.
   - Run the same with `LOG_LEVEL=DEBUG` and confirm the verbose detail (chunk scores,
     sub-questions) appears.
   - Confirm the UI's "How this answer was produced" expander is unchanged (trace intact).
   - Confirm the full offline test suite still passes (no regression from the `chat()`
     signature addition or the node logging calls).

## Out of scope

- Persistent log files / rotation (this is console-only; a file handler could be added
  later behind the same `setup_logging()` if wanted).
- Per-node wall-clock timing for non-LLM nodes (only the LLM call is timed, since it
  dominates; FAISS search is sub-millisecond and not worth timing).
- Request IDs / structured JSON logs / log aggregation (overkill for a local single-user
  demo).
- Any change to the in-UI trace expander's content.
