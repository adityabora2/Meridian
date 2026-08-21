"""The evidence contract shared by `generate` and `verify`.

These two nodes MUST agree on three things or citations silently break:

1. **Which chunks are evidence** -- `cap_pool` selects them, and both nodes call
   it on the same `state["retrieved"]`.
2. **How evidence is numbered** -- `format_evidence` assigns `[1]..[n]`, and the
   answer's markers resolve against that exact numbering.
3. **What counts as a citation marker** -- `parse_markers` is the one parser, so
   a format `generate` credits can never be a format `verify` rejects.

Before this module existed, (2) and (3) were duplicated in both nodes and kept
in step by comment. They drifted: the model emits ranges (`[1-7,9]`, `[5-12]`)
and both copies used `r"\\[(\\d+)\\]"`, which matches only bare single markers --
so correctly-grounded answers resolved zero citations and were then failed by
verify for *having no citations*. `parse_markers` accepts ranges and lists.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from meridian import config


# ---------------------------------------------------------------- pool control

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


# ------------------------------------------------------------ evidence framing

def format_evidence(retrieved: list[dict]) -> str:
    """Renders the evidence block as `[i] (document pN · heading) text`.

    The `[i]` here is the numbering the answer's citation markers refer to, so
    this function is the definition of that contract -- not a display detail.
    """
    lines = []
    for i, c in enumerate(retrieved, start=1):
        loc = f"{c['document_name']} p{c['page_number']}"
        if c.get("section_heading"):
            loc += f" · {c['section_heading']}"
        lines.append(f"[{i}] ({loc}) {c['text']}")
    return "\n\n".join(lines)


# ------------------------------------------------------------ citation parsing

# Any bracketed group; the contents are classified below rather than matched by
# one monolithic pattern, so prose brackets ("[sic]") stay ignorable while
# citation-shaped groups that don't parse are reported as malformed.
_BRACKET_RE = re.compile(r"\[([^\[\]]*)\]")

# A well-formed group: comma-separated single indices and/or ranges.
# Accepts "1", "1,2", "1-7,9", "1, 2", "1 - 7" and en-dash ranges.
_GROUP_RE = re.compile(r"^\s*\d+\s*(?:[-–]\s*\d+\s*)?(?:,\s*\d+\s*(?:[-–]\s*\d+\s*)?)*$")

# The placeholder the model emits when it copies the prompt's "[n]" literally
# instead of substituting a real index.
_PLACEHOLDER_RE = re.compile(r"^\s*n\d*\s*$", re.IGNORECASE)

_PART_RE = re.compile(r"(\d+)\s*(?:[-–]\s*(\d+))?")

# A range wider than this is a model hallucination, not a citation; expanding it
# would be pointless work, so it is reported as malformed instead.
_MAX_RANGE_WIDTH = 1000


@dataclass(frozen=True)
class Markers:
    """Everything both nodes need to know about an answer's citation markers."""

    indices: tuple[int, ...]        # in-range, sorted, de-duplicated
    out_of_range: tuple[int, ...]   # parsed cleanly but outside 1..n_evidence
    malformed: bool                 # a citation-shaped group that doesn't parse

    @property
    def any_found(self) -> bool:
        return bool(self.indices or self.out_of_range)


def parse_markers(answer: str, n_evidence: int) -> Markers:
    """Extracts citation indices from an answer.

    Understands the formats the model actually produces: `[3]`, `[1,2]`,
    `[1-7,9]`, `[5 - 12]`. Non-citation brackets are ignored; a group that looks
    like a citation but cannot be parsed (`[n]`, `[n2]`, `[1-]`, `[7-1]`) sets
    `malformed`, which callers treat as a hard failure.
    """
    found: set[int] = set()
    out: set[int] = set()
    malformed = False

    for raw in _BRACKET_RE.findall(answer):
        if _PLACEHOLDER_RE.match(raw):
            malformed = True
            continue
        if not _GROUP_RE.match(raw):
            # Only complain about groups that *start* like a citation; plain
            # prose brackets such as "[sic]" or "[see above]" are not markers.
            if re.match(r"^\s*\d", raw):
                malformed = True
            continue
        for start_s, end_s in _PART_RE.findall(raw):
            start = int(start_s)
            end = int(end_s) if end_s else start
            if end < start or end - start > _MAX_RANGE_WIDTH:
                malformed = True
                continue
            for i in range(start, end + 1):
                (found if 1 <= i <= n_evidence else out).add(i)

    return Markers(
        indices=tuple(sorted(found)),
        out_of_range=tuple(sorted(out)),
        malformed=malformed,
    )


def strip_markers(text: str) -> str:
    """Removes every bracketed citation group from prose.

    Used before number-checking an answer (a marker index is not a numeric claim)
    and before turning an unsupported claim into a search query. It must stay
    broader than a single-index pattern: with ranges accepted, leaving "[1-7,9]"
    in place would feed 1, 7 and 9 to the fabrication check as claimed figures.
    """
    return _BRACKET_RE.sub(" ", text)


def resolve_citations(answer: str, retrieved: list[dict]) -> list[dict]:
    """Maps an answer's in-range markers onto the evidence chunks they cite."""
    markers = parse_markers(answer, len(retrieved))
    citations = []
    for i in markers.indices:
        c = retrieved[i - 1]
        citations.append(
            {
                "marker": i,
                "chunk_id": c["chunk_id"],
                "document_name": c["document_name"],
                "page_number": c["page_number"],
                "section_heading": c.get("section_heading", ""),
                "score": c.get("score"),
            }
        )
    return citations
