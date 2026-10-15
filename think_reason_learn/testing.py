"""Fit and check models offline with scripted fake models.

Example:
    >>> from think_reason_learn.testing import FakeLLM, fake_llms
    >>> writer = FakeLLM(generate=lambda request: "Is the founder technical?")
    >>> judge = FakeLLM(answer=lambda sample, question: 0.9, kind="decision")
    >>> with fake_llms(writer=writer, judge=judge):
    ...     ...  # models now accept "fake:writer" and "fake:judge"
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Final, Literal

from pydantic import BaseModel

from think_reason_learn.llm import _spec
from think_reason_learn.llm._adapters import (
    ADAPTERS,
    Answers,
    GenerateRequest,
    ProviderError,
    Reply,
)
from think_reason_learn.llm._adapters._base import Failure as Failure_
from think_reason_learn.llm._questions import Answer, Question, YesNoQuestion
from think_reason_learn.llm._spec import Endpoint, ModelKind
from think_reason_learn.llm._usage import TokenCounts

type Failure = Literal[
    "rate_limit", "server_error", "timeout", "auth", "quota", "bad_request", "empty"
]
"""What a fake call does instead of replying."""

type GenerateReply = str | BaseModel | dict[str, object] | None
type AnswerReply = str | float | None


@dataclass(frozen=True, slots=True)
class FakeCall:
    """One call a fake model received."""

    kind: Literal["generate", "answer"]
    prompt: str | None = None
    instructions: str | None = None
    schema: type[BaseModel] | None = None
    sample: str | None = None
    questions: tuple[str, ...] = ()


@dataclass(slots=True)
class FakeLLM:
    """A scripted model.

    Args:
        generate: Reply to a generation request: text, or for a structured
            request an instance of its schema (or a dict to validate).
        answer: Reply to a question about a sample: a label, or the probability
            of yes for a yes/no question.
        kind: ``"reasoning"`` (writes and answers) or ``"decision"`` (answers
            with probabilities).
        failures: Failures to give, one per call, before any reply.
        tokens: Input and output tokens reported for each call.
    """

    generate: Callable[[GenerateRequest], GenerateReply] | None = None
    answer: Callable[[str, Question], AnswerReply] | None = None
    kind: ModelKind = "reasoning"
    failures: list[Failure] = field(default_factory=list)
    tokens: tuple[int, int] = (10, 5)
    calls: list[FakeCall] = field(default_factory=list)

    @property
    def n_calls(self) -> int:
        """Calls received, failed ones included."""
        return len(self.calls)


@contextmanager
def fake_llms(**models: FakeLLM) -> Iterator[None]:
    """Make each fake callable as ``"fake:<name>"`` inside the block."""
    ADAPTERS["fake"] = _ADAPTER
    patterns = {
        f"fake:{name}" for name, model in models.items() if model.kind == "decision"
    }
    _registry.update(models)
    _spec._DECISION_MODELS.update(patterns)
    try:
        yield
    finally:
        for name in models:
            _registry.pop(name, None)
        _spec._DECISION_MODELS.difference_update(patterns)


_registry: Final[dict[str, FakeLLM]] = {}

_ERRORS: Final[dict[Failure, tuple[Failure_, str]]] = {
    "rate_limit": ("rate_limit", "fake rate limit"),
    "server_error": ("transient", "fake server error"),
    "timeout": ("transient", "fake timeout"),
    "auth": ("auth", "fake key rejected"),
    "quota": ("quota", "fake insufficient_quota"),
    "bad_request": ("bad_request", "fake bad request"),
}


class _FakeAdapter:
    def max_questions(self, endpoint: Endpoint) -> int:
        return (
            50 if endpoint.kind == "decision" else endpoint.settings.questions_per_call
        )

    async def generate(
        self, endpoint: Endpoint, request: GenerateRequest, dropped: frozenset[str]
    ) -> Reply[str | BaseModel | None]:
        fake = _fake(endpoint)
        fake.calls.append(
            FakeCall("generate", request.prompt, request.instructions, request.schema)
        )
        if _failed(fake) or fake.generate is None:
            return Reply(None, _tokens(fake))
        output = fake.generate(request)
        if isinstance(output, dict) and request.schema is not None:
            output = request.schema.model_validate(output)
        return Reply(
            output if not isinstance(output, dict) else None,
            _tokens(fake),
            endpoint.model_id,
        )

    async def answer(
        self,
        endpoint: Endpoint,
        sample: str,
        questions: Mapping[str, Question],
        dropped: frozenset[str],
    ) -> Reply[Answers]:
        fake = _fake(endpoint)
        texts = tuple(q.text for q in questions.values())
        fake.calls.append(FakeCall("answer", sample=sample, questions=texts))
        if _failed(fake) or fake.answer is None:
            return Reply(dict.fromkeys(questions), _tokens(fake))
        answers: Answers = {
            qid: _answer(fake.answer(sample, q), q, decision=fake.kind == "decision")
            for qid, q in questions.items()
        }
        return Reply(answers, _tokens(fake), endpoint.model_id)


_ADAPTER: Final = _FakeAdapter()


def _fake(endpoint: Endpoint) -> FakeLLM:
    if endpoint.model_id not in _registry:
        raise ProviderError(
            "bad_request", f"No fake model {endpoint.name} in fake_llms(...)."
        )
    return _registry[endpoint.model_id]


def _failed(fake: FakeLLM) -> bool:
    """Give the next scripted failure: raise it, or return True for "empty"."""
    if not fake.failures:
        return False
    failure = fake.failures.pop(0)
    if failure == "empty":
        return True
    kind, message = _ERRORS[failure]
    raise ProviderError(kind, message, retry_after=0.0)


def _tokens(fake: FakeLLM) -> TokenCounts:
    return TokenCounts(input=fake.tokens[0], output=fake.tokens[1])


def _answer(reply: AnswerReply, question: Question, *, decision: bool) -> Answer | None:
    if isinstance(reply, (int, float)) and isinstance(question, YesNoQuestion):
        if decision:
            return Answer.yes_probability(float(reply))
        reply = "YES" if reply >= 0.5 else "NO"
    if isinstance(reply, str) and reply in question.labels:
        return Answer.certain(reply, question.labels)
    return None


def scripted(
    replies: Iterable[GenerateReply],
) -> Callable[[GenerateRequest], GenerateReply]:
    """A ``generate`` that gives ``replies`` in order, then repeats the last."""
    queue = list(replies)

    def reply(request: GenerateRequest) -> GenerateReply:
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return reply


__all__ = ["FakeCall", "FakeLLM", "Failure", "GenerateRequest", "fake_llms", "scripted"]
