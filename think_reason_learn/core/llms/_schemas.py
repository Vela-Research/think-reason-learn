from typing import TypeVar, Tuple, Generic, List, TypeAlias, Literal, override, Union
import math
from dataclasses import dataclass

from pydantic import BaseModel

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
    """A response from an LLM.

    Attributes:
        response: The response from the LLM.
        logprobs: The log probabilities of the response.
        total_tokens: The total number of tokens in the response.
        provider_model: The provider and model used to generate the response.

    Properties:
        average_confidence: The average confidence of the response calculated
        from the log probabilities.
    """

    response: T | None = None
    logprobs: List[Tuple[str, float | None]]
    total_tokens: int | None = None
    provider_model: LLMChoiceModel

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

    For example:

    ```py
    def get(timeout: Union[int, TRLNotGiven, None] = TRLNotGiven()) -> Response: ...


    get(timeout=1)  # 1s timeout
    get(timeout=None)  # No timeout
    get()  # Default timeout behavior, which may not be statically known at the method
    definition.
    ```
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
    """Tokens used in an LLM call.

    Attributes:
        provider: The provider of the LLM.
        model: The LLM model used.
        value: The token count.
    """

    provider: LLMProvider
    model: LLMChatModel
    value: int | None = None
