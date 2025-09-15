from __future__ import annotations

from typing import Dict, Literal

from think_reason_learn.core._types import JSONValue


Sample = Dict[str, JSONValue]
QuestionType = Literal["INFERENCE", "CODE"]
Criterion = Literal["gini"]
