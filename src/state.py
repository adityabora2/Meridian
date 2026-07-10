from __future__ import annotations

from typing import TypedDict


class RAGState(TypedDict, total=False):
    question: str

    route: str
    mode_label: str

    sub_questions: list[str]

    retrieved: list[dict]

    answer: str
    citations: list[dict]

    critique_clean: bool
    unsupported_claims: list[str]
    iterations: int
    failure_type: str
    verify_feedback: str
    heal_action: str
    retry_queries: list[str]
    verification_warnings: list[str]

    trace: list[str]
