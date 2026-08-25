"""Deterministic document-name matching: does this text name an indexed document?

The router's two guardrails and search's per-document scoping all key off these
functions, so they run several times per user question. Every derived structure
(the corpus blob map, per-document token sets, query token sets) is therefore
memoized -- see the cache notes on each.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

from meridian.ingest.store import load_index


@lru_cache(maxsize=1)
def _corpus_from_index() -> dict[str, str]:
    """Walks the full metadata list once to map document_name -> matchable blob
    (title + filename stem). Cached: the walk is O(all chunks) -- thousands of
    them -- and the result only changes when the index is rebuilt."""
    _, metadata = load_index()
    corpus: dict[str, str] = {}
    for chunk in metadata:
        if chunk.document_name not in corpus:
            stem = Path(chunk.document_name).stem
            corpus[chunk.document_name] = f"{chunk.document_title} {stem}"
    return corpus


def _document_match_corpus() -> dict[str, str]:
    """Maps document_name -> a matchable text blob (title + filename stem) for
    every document currently in the loaded index. Kept as a thin indirection so
    tests can substitute a synthetic corpus without touching the index."""
    return _corpus_from_index()


_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Common English words that appear incidentally in paper titles/filenames
# (e.g. "Attention Is All You Need", "gpt2.pdf" vs. a query asking "what
# is..."). Left in, these produce false ties/false positives between
# unrelated documents; filtering them out sharpens matching on the
# distinctive, document-specific vocabulary that actually identifies a paper.
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "and", "or", "but", "not",
    "no", "do", "does", "did", "how", "what", "which", "who", "whom",
    "this", "that", "these", "those", "it", "its", "as", "by", "from",
    "about", "into", "use", "uses", "used", "using", "can", "will",
    "would", "should", "could", "i", "you", "he", "she", "we", "they",
    "them", "their", "our", "your", "my",
})


# Matches a run of letters/digits allowing internal hyphens (e.g. "gpt-2",
# "text-to-text"), so hyphenated model names survive as a single unit that can
# be normalized to match an unhyphenated filename stem like "gpt2".
_HYPHEN_RUN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)+")


@lru_cache(maxsize=1024)
def _tokenize(text: str) -> frozenset[str]:
    """Cached because the same question is tokenized repeatedly (once per
    document in match_document, again in implicated_documents), and document
    blobs re-tokenize identically on every query. Returns a frozenset so a
    cached value can never be mutated by a caller."""
    text = text.lower()
    tokens = {t for t in _TOKEN_RE.findall(text) if t not in _STOPWORDS}
    # Add de-hyphenated variants of hyphenated runs so "GPT-2" (which
    # otherwise splits into {"gpt", "2"}) also yields "gpt2", matching a
    # filename stem like "gpt2.pdf". This closes a recall gap where hyphenated
    # model names never matched their own unhyphenated document.
    for run in _HYPHEN_RUN_RE.findall(text):
        joined = run.replace("-", "")
        if joined not in _STOPWORDS:
            tokens.add(joined)
    return frozenset(tokens)


def _stem_tokens(document_name: str) -> frozenset[str]:
    return _tokenize(Path(document_name).stem.replace("_", " "))


def _score_document_match(query_tokens: set[str], doc_tokens: set[str]) -> int:
    return len(query_tokens & doc_tokens)


# Extra weight given to query tokens that match a document's own filename
# stem (e.g. "bert" in "bert.pdf"). Title-word overlap alone is too weak a
# signal in a corpus of vocabulary-similar sibling papers -- e.g. ALBERT's
# title ("A LITE BERT FOR...") literally contains the word "BERT", so a
# BERT-focused query ties bert.pdf and albert.pdf on title words alone. The
# filename stem is the most direct, unambiguous per-document label, so a
# query naming it should count far more than an incidental shared title word.
_STEM_MATCH_WEIGHT = 3

_MATCH_MARGIN = 2


def match_document(text: str) -> Optional[str]:
    """The single document this text names, or None if ambiguous or none."""
    corpus = _document_match_corpus()
    if not corpus:
        return None

    query_tokens = _tokenize(text)
    scores: dict[str, int] = {}
    for name, doc_text in corpus.items():
        base = _score_document_match(query_tokens, _tokenize(doc_text))
        stem_overlap = _score_document_match(query_tokens, _stem_tokens(name))
        scores[name] = base + _STEM_MATCH_WEIGHT * stem_overlap

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if not ranked or ranked[0][1] == 0:
        return None
    if len(ranked) == 1:
        return ranked[0][0]

    best_name, best_score = ranked[0]
    second_score = ranked[1][1]
    if best_score - second_score >= _MATCH_MARGIN:
        return best_name
    return None


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
    return sorted(name for name in corpus if query_tokens & _stem_tokens(name))
