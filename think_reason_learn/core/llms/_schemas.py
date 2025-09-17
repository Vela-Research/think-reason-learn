from typing import TypeVar, Tuple, Generic, List, TypeAlias, Literal, override, Union
import math
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from ._anthropic.schemas import (
    AnthropicChatModel,
    AnthropicChoice,
    AnthropicChoiceDict,
    NOT_GIVEN as ANTHROPIC_NOT_GIVEN,
    NotGiven as AnthropicNotGiven,
)
from ._openai.schemas import (
    OpenAIChatModel,
    OpenAIChoice,
    OpenAIChoiceDict,
    NotGiven as OpenAINotGiven,
    NOT_GIVEN as OPENAI_NOT_GIVEN,
)
from ._google.schemas import GoogleChatModel, GoogleChoice, GoogleChoiceDict
from ._xai.schemas import xAIChatModel, XAIChoice, XAIChoiceDict


T = TypeVar("T", bound=BaseModel | str, covariant=True)


LLMProvider: TypeAlias = Literal["anthropic", "google", "openai", "xai"]
LLMChatModel: TypeAlias = Union[
    AnthropicChatModel,
    GoogleChatModel,
    OpenAIChatModel,
    xAIChatModel,
]
LLMChoiceModel: TypeAlias = Union[
    AnthropicChoice,
    GoogleChoice,
    OpenAIChoice,
    XAIChoice,
]
LLMChoiceDict: TypeAlias = Union[
    AnthropicChoiceDict,
    GoogleChoiceDict,
    OpenAIChoiceDict,
    XAIChoiceDict,
]
LLMChoice: TypeAlias = LLMChoiceModel | LLMChoiceDict


class LLMResponse(BaseModel, Generic[T]):
    """A response from an LLM."""

    response: T | None = Field(default=None, description="The response from the LLM.")
    logprobs: List[Tuple[str, float | None]] = Field(
        description="The log probabilities of the response."
    )
    total_tokens: int | None = Field(
        default=None,
        description="The total number of input and output tokens used "
        "to generate the response.",
    )
    provider_model: LLMChoiceModel = Field(
        description="The provider and model used to generate the response."
    )

    @property
    def average_confidence(self) -> float | None:
        if not self.logprobs:
            return

        lps = [lp[1] for lp in self.logprobs if lp[1] is not None]
        if not lps:
            return

        ave_lp = sum(lps) / len(lps)
        return math.exp(ave_lp)


class NotGiven:
    """A sentinel singleton class used to distinguish omitted keyword arguments.

    Examples:
        .. code-block:: python

            def get(timeout: int | NotGiven | None = NotGiven()) -> Response: ...

            get(timeout=1)      # 1s timeout
            get(timeout=None)   # No timeout
            get()               # Default timeout behavior; may not be statically
                                # known at the method definition.
    """

    ANTHROPIC_NOT_GIVEN: AnthropicNotGiven = ANTHROPIC_NOT_GIVEN
    OPENAI_NOT_GIVEN: OpenAINotGiven = OPENAI_NOT_GIVEN

    def __bool__(self) -> Literal[False]:
        return False

    @override
    def __repr__(self) -> str:
        return "NOT_GIVEN"


NOT_GIVEN = NotGiven()


@dataclass(slots=True)
class TokenCount:
    """A token count from an LLM."""

    provider: LLMProvider = field(metadata={"description": "The provider of the LLM."})
    model: LLMChatModel = field(metadata={"description": "The LLM model used."})
    value: int | None = field(
        default=None, metadata={"description": "The token count."}
    )
