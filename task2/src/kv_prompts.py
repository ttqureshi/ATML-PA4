"""Prompt splitting for the retrieval cases.

The question suffix starts at the ``QUESTION_BEGIN`` marker in ``user_text``.
Gold codes and evidence character spans are not read here.
"""

from __future__ import annotations

QUESTION_MARKER = "QUESTION_BEGIN"


def split_user_text(user_text: str) -> tuple[str, str]:
    at = user_text.rfind(QUESTION_MARKER)
    if at < 0:
        raise ValueError("user text has no QUESTION_BEGIN marker")
    return user_text[:at], user_text[at:]


def question_char_span(prompt: str, user_text: str) -> tuple[int, int]:
    """Character span of the question suffix inside an already rendered prompt string."""
    context, _question = split_user_text(user_text)
    base = prompt.rfind(user_text)
    if base < 0:
        marker = prompt.rfind(QUESTION_MARKER)
        if marker < 0:
            raise ValueError("rendered prompt has no QUESTION_BEGIN marker")
        return marker, len(prompt)
    return base + len(context), base + len(user_text)


def tokens_overlapping(offsets: list[tuple[int, int]], start: int, end: int) -> list[int]:
    """Token indexes whose character span overlaps ``[start, end)``. Empty spans are skipped."""
    chosen = []
    for index, (left, right) in enumerate(offsets):
        if right <= left:
            continue
        if right <= start or left >= end:
            continue
        chosen.append(index)
    return chosen


def policy_fields(row: dict) -> dict:
    """Fields a retention policy may see. The gold code and evidence spans are omitted."""
    allowed = ("id", "user_text", "position", "distractors", "target_tokens")
    return {key: row[key] for key in allowed if key in row}
