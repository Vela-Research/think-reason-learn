from typing import TypeVar, Tuple, Generic, List
import math

from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel | str)


class LLMResponse(BaseModel, Generic[T]):
    response: T
    logprobs: List[Tuple[str, float | None]]
    total_tokens: int | None = None

    @property
    def average_confidence(self) -> float | None:
        if not self.logprobs:
            return

        lps = [lp[1] for lp in self.logprobs if lp[1] is not None]
        if not lps:
            return

        ave_lp = sum(lps) / len(lps)
        return math.exp(ave_lp)
