from typing import Annotated, List, Literal, NotRequired, Sequence, TypeAlias
from typing import TypedDict

from pydantic import BaseModel, Field


JevModel: TypeAlias = str | Literal["jev-latest"]


class JevChoice(BaseModel):
    """Jev, Typesafe's System One model, for answering questions about samples.

    Jev answers questions; it does not generate text, so use it for answering
    (``qanswer_llmc`` / ``predict_llmc``) and a chat model for generation.
    Requires ``TYPESAFE_API_KEY``.

    Args:
        model: The Jev model. ``"jev-latest"`` follows Typesafe's newest release.
        max_cost_usd: Cost cap for one ``fit()`` or ``predict()`` run. The
            estimated cost is printed before anything is sent, and the run stops
            if the estimate or the actual spend passes the cap.
        cache: Reuse answers already paid for. Answers are stored in
            ``~/.cache/think_reason_learn/jev`` (or ``$XDG_CACHE_HOME``), keyed by
            model, sample and questions. Set ``False`` to always ask Jev again.
    """

    provider: Literal["jev"] = "jev"
    model: JevModel = "jev-latest"
    max_cost_usd: float = Field(default=10.0, gt=0)
    cache: bool = True


class JevChoiceDict(TypedDict):
    """Jev, Typesafe's System One model, for answering questions about samples."""

    provider: Literal["jev"]
    model: JevModel
    max_cost_usd: NotRequired[float]
    cache: NotRequired[bool]


class NoulQuestion(BaseModel):
    """A yes/no question. Jev answers with the probability of yes, in [0, 1]."""

    type: Literal["noul"] = "noul"
    instructions: str


class ChoiceQuestion(BaseModel):
    """A multiple-choice question. Jev answers with exactly one of ``labels``."""

    type: Literal["choice"] = "choice"
    instructions: str
    labels: List[str] = Field(min_length=2)


JevQuestion: TypeAlias = Annotated[
    NoulQuestion | ChoiceQuestion, Field(discriminator="type")
]
AnswerValue: TypeAlias = float | str
"""A noul probability (float) or a choice label (str)."""


def is_jev(choice: object) -> bool:
    """Whether an LLM choice (model or dict) is Jev."""
    if isinstance(choice, dict):
        return choice.get("provider") == "jev"
    return getattr(choice, "provider", None) == "jev"


def first_jev_choice(llm_priority: Sequence[object]) -> JevChoice | None:
    """The first Jev choice in a priority list, as a ``JevChoice``."""
    for choice in llm_priority:
        if isinstance(choice, dict) and is_jev(choice):
            return JevChoice(**choice)
        if isinstance(choice, JevChoice):
            return choice
    return None
