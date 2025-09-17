from typing import TypeAlias, Literal, TypedDict

from pydantic import BaseModel


GoogleChatModel: TypeAlias = (
    str
    | Literal[
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
    ]
)


class GoogleChoice(BaseModel):
    """An LLM from Google."""

    provider: Literal["google"] = "google"
    model: GoogleChatModel


class GoogleChoiceDict(TypedDict):
    """An LLM from Google."""

    provider: Literal["google"]
    model: GoogleChatModel
