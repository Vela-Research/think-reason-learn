"""Answering questions with a reasoning model, by prompting for a label.

The sample goes first and the question last, with one schema per label set, so
a provider's prompt cache can reuse the sample across questions. The answer is
the label the model picked, with probability 1.
"""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Final

from pydantic import BaseModel, Field, create_model

from .._questions import Answer, Question
from .._spec import Endpoint
from ._base import Answers, GenerateRequest, Reply

ANSWER_INSTRUCTIONS: Final = (
    "You answer questions about one sample. Base every answer only on the "
    "sample, and reply in the requested format."
)

type Generate = Callable[
    [Endpoint, GenerateRequest, frozenset[str]],
    Awaitable[Reply[str | BaseModel | None]],
]


class ReasoningAdapter(ABC):
    """What every reasoning-model adapter shares.

    Questions per call follow the model's settings, and answers are given by
    prompting for a label.
    """

    def max_questions(self, endpoint: Endpoint) -> int:
        """Questions per call, as the model's settings ask."""
        return endpoint.settings.questions_per_call

    @abstractmethod
    async def generate(
        self, endpoint: Endpoint, request: GenerateRequest, dropped: frozenset[str]
    ) -> Reply[str | BaseModel | None]:
        """Text, or an instance of ``request.schema``; ``None`` if unusable."""

    async def answer(
        self,
        endpoint: Endpoint,
        sample: str,
        questions: Mapping[str, Question],
        dropped: frozenset[str],
    ) -> Reply[Answers]:
        """Answers by prompting for a label."""
        return await prompted_answer(
            self.generate, endpoint, sample, questions, dropped
        )


async def prompted_answer(
    generate: Generate,
    endpoint: Endpoint,
    sample: str,
    questions: Mapping[str, Question],
    dropped: frozenset[str],
) -> Reply[Answers]:
    """Ask a reasoning model ``questions`` about ``sample`` in one call."""
    ids = list(questions)
    asked = [questions[qid] for qid in ids]
    request = GenerateRequest(
        prompt=_prompt(asked),
        instructions=ANSWER_INSTRUCTIONS,
        schema=_schema(tuple(q.labels for q in asked)),
        prefix=f"Sample:\n{sample}",
    )
    reply = await generate(endpoint, request, dropped)
    picked = _picked(reply.output, len(ids))
    answers: Answers = {
        qid: Answer.certain(label, q.labels) if label in q.labels else None
        for qid, q, label in zip(ids, asked, picked, strict=True)
    }
    return Reply(
        answers, reply.tokens, reply.served_model, reply.provider_cost, reply.headers
    )


def _prompt(questions: Sequence[Question]) -> str:
    if len(questions) == 1:
        [question] = questions
        options = ", ".join(question.labels)
        return f"Question: {question.text}\nAnswer with exactly one of: {options}."
    numbered = "\n".join(
        f"{i}. {q.text} (one of: {', '.join(q.labels)})"
        for i, q in enumerate(questions, 1)
    )
    return (
        f"Questions:\n{numbered}\nAnswer question 1 in a1, question 2 in a2, and so on."
    )


@functools.cache
def _schema(labels: tuple[tuple[str, ...], ...]) -> type[BaseModel]:
    fields = {
        f"a{i}": (str, Field(json_schema_extra={"enum": list(options)}))
        for i, options in enumerate(labels, 1)
    }
    # pydantic types **field_definitions loosely; the fields are (type, Field).
    model: object = create_model("Answers", **fields)  # pyright: ignore[reportCallIssue, reportArgumentType, reportUnknownVariableType]
    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        raise TypeError("create_model did not return a model.")
    return model


def _picked(output: str | BaseModel | None, count: int) -> list[str]:
    if not isinstance(output, BaseModel):
        return [""] * count
    return [str(getattr(output, f"a{i}", "")) for i in range(1, count + 1)]
