from __future__ import annotations

from typing import TypeVar, Tuple, Generic, List, TypeAlias, Literal
from typing import Dict, override, Union, Any
import math
from dataclasses import dataclass, field, asdict
import asyncio

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
    number_of_calls: int = field(
        default=0, metadata={"description": "The number of calls to the LLM."}
    )
    value: int = field(default=0, metadata={"description": "The token count."})
    is_min_estimate: bool = field(
        default=False,
        metadata={
            "description": "Whether the token count is a minimum estimate. "
            "Will be True if the some calls to this LLM could not return "
            "the actual token count."
        },
    )
    callers: Dict[str, int] = field(
        default_factory=dict,
        metadata={
            "description": "The callers (functions/modules) and their number "
            "of calls to the LLM."
        },
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TokenCount:
        return cls(**d)


@dataclass(slots=True)
class TokenCounter:
    """Compactly represents token usage across multiple calls to LLMs."""

    token_counts: Dict[str, TokenCount] = field(
        default_factory=dict,
        metadata={"description": "The token counts for each provider/model pair."},
    )
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def append(
        self,
        model: LLMChatModel,
        provider: LLMProvider,
        value: int | None,
        caller: str,
    ) -> TokenCount:
        """Append a token count to the counter.

        Args:
            model: The model used.
            provider: The provider of the model.
            value: The token count.
            caller: The caller (function/module) of the LLM.
                Preferably the function name.

        Returns:
            The token count object.
        """
        key = f"{provider}/{model}"
        async with self._lock:
            token_count = self.token_counts.get(
                key, TokenCount(provider=provider, model=model)
            )
            token_count.number_of_calls += 1
            token_count.value += value if value is not None else 0
            token_count.callers[caller] = token_count.callers.get(caller, 0) + 1
            if not token_count.is_min_estimate and value is None:
                token_count.is_min_estimate = True
            self.token_counts[key] = token_count
            return token_count

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("_lock", None)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TokenCounter:
        token_counts = {
            k: TokenCount.from_dict(v) for k, v in d["token_counts"].items()
        }
        self = cls(token_counts=token_counts)
        return self
