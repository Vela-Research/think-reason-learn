from __future__ import annotations
from typing import Union, Dict, List

JSONScalar = Union[str, int, float, bool, None]
JSONValue = Union[JSONScalar, Dict[str, "JSONValue"], List["JSONValue"]]
