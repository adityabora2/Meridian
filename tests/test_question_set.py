"""Offline structural checks on the live harness's question set.

`tests/test_questions.py` needs Ollama and is never collected by pytest (it has
no test_* functions and is excluded from CI runs). But its question set has
properties that CAN be checked offline against the real index, and that silently
corrupt the harness's numbers when they break:

  - a "hard" question naming only ONE indexed document is downgraded to medium
    by the router's guard, so it is graded a routing MISS that is really a
    question-wording bug;
  - an "easy" question naming an indexed document is upgraded to medium, same
    problem in the other direction;
  - a "medium" question naming NO resolvable document silently loses document
    scoping, so Mode-2 retrieval is corpus-wide rather than single-hop.

These tests catch all three without an LLM: they exercise only the deterministic
matching functions the router's guards call. They also fail loudly if the index
is re-ingested with a different corpus and the questions are not updated -- the
exact drift that made the previous NLP-paper question set meaningless against
the current cross-domain corpus.
"""

from __future__ import annotations

import pytest

from meridian.ingest import implicated_documents, match_document
from tests.test_questions import QUESTIONS

EASY = [q for q, r in QUESTIONS if r == "easy"]
MEDIUM = [q for q, r in QUESTIONS if r == "medium"]
HARD = [q for q, r in QUESTIONS if r == "hard"]


def test_question_set_shape():
    assert len(QUESTIONS) == 30
    assert (len(EASY), len(MEDIUM), len(HARD)) == (10, 10, 10)


def test_every_expected_route_is_valid():
    assert {r for _, r in QUESTIONS} == {"easy", "medium", "hard"}


def test_no_duplicate_questions():
    questions = [q for q, _ in QUESTIONS]
    assert len(set(questions)) == len(questions)


@pytest.mark.parametrize("question", EASY)
def test_easy_questions_name_no_indexed_document(question):
    """An easy question that names a document is upgraded to medium by the
    router (router.py: easy + match_document -> medium), which would make the
    'easy' label unreachable by design rather than a real routing result."""
    matched = match_document(question)
    assert matched is None, (
        f"easy question resolves to {matched!r}; the router would upgrade it to "
        f"medium, so the 'easy' label can never be hit: {question!r}"
    )


@pytest.mark.parametrize("question", MEDIUM)
def test_medium_questions_name_exactly_one_document(question):
    """Medium = single-hop grounded in one document. If match_document returns
    None the search is never scoped to that document, so the question is not
    actually testing the single-hop path."""
    matched = match_document(question)
    assert matched is not None, (
        f"medium question resolves to no document, so retrieval will not be "
        f"scoped and the single-hop path is untested: {question!r}"
    )


@pytest.mark.parametrize("question", HARD)
def test_hard_questions_implicate_multiple_documents(question):
    """The router downgrades hard -> medium when exactly ONE document is
    implicated by filename stem (router.py's downgrade guard). A hard question
    implicating one document is therefore graded a MISS for a reason that has
    nothing to do with the router's judgment."""
    implicated = implicated_documents(question)
    assert len(implicated) != 1, (
        f"hard question implicates exactly one document ({implicated[0]!r}); the "
        f"router's downgrade guard will route it to medium and the harness will "
        f"score it as a routing miss: {question!r}"
    )


@pytest.mark.parametrize("question", HARD)
def test_hard_questions_are_answerable_from_the_corpus(question):
    """0 implicated documents stays hard (the router fails safe toward more
    retrieval), so it would not show up as a routing miss -- but a cross-document
    question whose documents aren't in the index produces the unsupportable-answer
    situation that made the old question set's citation metric meaningless."""
    implicated = implicated_documents(question)
    assert len(implicated) >= 2, (
        f"hard question implicates {len(implicated)} indexed documents; a "
        f"cross-document comparison needs at least 2 present in the corpus for "
        f"the Mode-3 citation metric to mean anything: {question!r}"
    )
