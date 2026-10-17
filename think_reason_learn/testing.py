"""Fit and check models offline with scripted fake models.

Example:
    >>> from think_reason_learn.testing import FakeLLM, fake_llms
    >>> writer = FakeLLM(generate=lambda request: "Is the review positive?")
    >>> judge = FakeLLM(answer=lambda sample, question: 0.9, kind="decision")
    >>> with fake_llms(writer=writer, judge=judge):
    ...     pass  # models now accept "fake:writer" and "fake:judge"
"""

from __future__ import annotations

import pickle
import uuid
from collections.abc import Callable, Generator, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel
from sklearn.exceptions import NotFittedError

from think_reason_learn.core._classifier import TRLClassifier
from think_reason_learn.core._data import LabelInput, TextInput
from think_reason_learn.core._sklearn import check_constructor, clone_estimator
from think_reason_learn.llm import (
    NO,
    YES,
    YES_AT,
    Answer,
    Answers,
    Endpoint,
    ModelKind,
    Price,
    Question,
    YesNoQuestion,
)
from think_reason_learn.llm._adapters import (
    ADAPTERS,
    GenerateRequest,
    ProviderError,
    Reply,
)
from think_reason_learn.llm._adapters._base import DecisionAdapter, Failure
from think_reason_learn.llm._spec import extra_provider
from think_reason_learn.llm._usage import TokenCounts

__all__ = [
    "FakeCall",
    "FakeLLM",
    "FakeFailure",
    "GenerateRequest",
    "check_classifier",
    "fake_llms",
    "scripted",
]

type FakeFailure = Literal[
    "rate_limit",
    "server_error",
    "timeout",
    "auth",
    "quota",
    "unavailable",
    "bad_request",
    "empty",
]
"""What a fake call does instead of replying."""

type GenerateReply = str | BaseModel | dict[str, object] | None
type AnswerReply = str | float | None


_registry: Final[dict[str, FakeLLM]] = {}

_ERRORS: Final[dict[FakeFailure, tuple[Failure, str]]] = {
    "rate_limit": ("rate_limit", "fake rate limit"),
    "server_error": ("transient", "fake server error"),
    "timeout": ("transient", "fake timeout"),
    "auth": ("auth", "fake key rejected"),
    "quota": ("quota", "fake insufficient_quota"),
    "unavailable": ("unavailable", "fake model not found"),
    "bad_request": ("bad_request", "fake bad request"),
}


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
    failures: list[FakeFailure] = field(default_factory=list[FakeFailure])
    tokens: tuple[int, int] = (10, 5)
    calls: list[FakeCall] = field(default_factory=list[FakeCall])

    @property
    def n_calls(self) -> int:
        """Calls received, failed ones included."""
        return len(self.calls)


@contextmanager
def fake_llms(**models: FakeLLM) -> Generator[None]:
    """Make each fake callable as ``"fake:<name>"`` inside the block."""
    ADAPTERS["fake"] = _ADAPTER
    _registry.update(models)
    # A fresh address per block, so replies cached by earlier fakes never answer.
    address = f"fake://{uuid.uuid4().hex}"
    try:
        # Fakes stand in for hosted models: priced from the catalog, else unknown.
        with extra_provider(
            "fake", "fake", address, catalog=_decision_fakes, local=False
        ):
            yield
    finally:
        for name in models:
            _registry.pop(name, None)


def scripted(
    replies: Iterable[GenerateReply],
) -> Callable[[GenerateRequest], GenerateReply]:
    """A ``generate`` that gives ``replies`` in order, then repeats the last."""
    queue = list(replies)

    def reply(request: GenerateRequest) -> GenerateReply:
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return reply


def check_classifier(
    make: Callable[[], TRLClassifier], X: TextInput, y: LabelInput, tmp_dir: Path
) -> None:
    """Check a classifier keeps the library's contract.

    It must behave as a scikit-learn classifier (settings stored untouched,
    ``clone``, ``get_params``, fitted attributes, prediction shapes), refit,
    pickle, and save and load without API keys. Run it inside ``fake_llms``.

    Args:
        make: Builds the classifier with fake models.
        X: Samples to fit on and predict.
        y: Their labels.
        tmp_dir: A folder to save the model to.
    """
    model = make()
    name = type(model).__name__
    check_constructor(name, make)
    type(model)()  # every setting has a default
    clone_estimator(model)
    try:
        model.predict(X)
    except NotFittedError:
        pass
    else:
        raise AssertionError("predict before fit must raise NotFittedError")
    assert model.fit(X, y) is model, "fit must return the model"
    predicted = _check_predictions(model, X, y)
    restored = pickle.loads(pickle.dumps(model))
    assert (restored.predict(X) == predicted).all(), "a pickle must predict the same"
    folder = model.save(tmp_dir / name)
    assert "api_key" not in (folder / "model.json").read_text()
    assert (type(model).load(folder).predict(X) == predicted).all()
    assert model.fit(X, y) is model, "a fitted model must refit"


def _check_predictions(
    model: TRLClassifier, X: TextInput, y: LabelInput
) -> npt.NDArray[np.generic]:
    """Probabilities and labels of the right shape; returns the labels."""
    proba = model.predict_proba(X)
    n_samples = len(X)
    assert proba.shape == (n_samples, len(model.classes_)), proba.shape
    assert np.allclose(proba.sum(axis=1), 1) and (proba >= 0).all(), proba
    predicted = model.predict(X)
    assert len(predicted) == n_samples and set(predicted) <= set(model.classes_)
    assert set(model.classes_.tolist()) == set(np.asarray(y).tolist()), model.classes_
    assert model.usage_.calls > 0, "model calls must be recorded in usage_"
    return predicted


class _FakeAdapter:
    def max_questions(self, endpoint: Endpoint) -> int:
        return (
            DecisionAdapter.max_questions_per_call
            if endpoint.kind == "decision"
            else endpoint.settings.questions_per_call
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


def _decision_fakes() -> dict[str, Price | None]:
    """The fake provider's catalog: fakes of kind ``"decision"``, unpriced."""
    return {name: None for name, fake in _registry.items() if fake.kind == "decision"}


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
        reply = YES if reply >= YES_AT else NO
    if isinstance(reply, str) and reply in question.labels:
        return Answer.certain(reply, question.labels)
    return None
