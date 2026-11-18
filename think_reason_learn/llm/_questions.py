"""Questions asked about a sample, and their answers."""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final

YES: Final = "YES"
NO: Final = "NO"
YES_AT: Final = 0.5
"""A probability of yes from which the answer counts as yes."""


@dataclass(frozen=True, slots=True)
class YesNoQuestion:
    """A yes/no question about a sample.

    Args:
        text: The question, e.g. ``"Does the email ask for a password?"``.
    """

    text: str

    @property
    def labels(self) -> tuple[str, str]:
        """The possible answers."""
        return (YES, NO)


@dataclass(frozen=True, slots=True)
class ChoiceQuestion:
    """A question answered with one of several labels.

    Args:
        text: The question.
        labels: The possible answers, at least two, all different.
        descriptions: What some or all labels mean, by label, e.g.
            ``{"seed": "before any priced round"}``. Every model is shown them:
            System One models in ``criteria``, OpenAI's Decisions API on each
            choice, and reasoning models in the prompt. A decision model reads
            each option through its description, so one trained or prompted with
            descriptions should be asked with the same ones. ``None`` (or an
            empty mapping) asks with the labels alone, as before.
    """

    text: str
    labels: tuple[str, ...]
    descriptions: Mapping[str, str] | None = field(default=None, hash=False)

    def __post_init__(self) -> None:
        """Check the labels and descriptions, and keep labels as a tuple."""
        object.__setattr__(self, "labels", tuple(self.labels))
        if len(self.labels) < 2 or len(set(self.labels)) != len(self.labels):
            raise ValueError(
                f"A choice question needs two or more different labels, got "
                f"{self.labels}."
            )
        if self.descriptions is None:
            return
        unknown = [label for label in self.descriptions if label not in self.labels]
        if unknown:
            raise ValueError(
                f"Descriptions are for the question's labels {self.labels}; "
                f"{unknown} are not among them."
            )
        described = {
            label: self.descriptions[label]
            for label in self.labels
            if label in self.descriptions
        }  # a copy in label order, so the wire and the cache key are stable
        object.__setattr__(self, "descriptions", described or None)


type Question = YesNoQuestion | ChoiceQuestion


@dataclass(frozen=True, slots=True)
class Answer:
    """One model's answer to one question about one sample.

    Args:
        label: The chosen label: ``"YES"``/``"NO"`` for a yes/no question.
        probabilities: Probability of each label. Decision models estimate
            them; a reasoning model's answer puts 1.0 on its label.
    """

    label: str
    probabilities: dict[str, float]

    @classmethod
    def certain(cls, label: str, labels: tuple[str, ...]) -> Answer:
        """An answer given without probabilities, as reasoning models give.

        One shared instance per label and labels: a large run holds millions.
        Its ``probabilities`` must not be changed.
        """
        return _certain(label, labels)

    @classmethod
    def yes_probability(cls, probability: float) -> Answer:
        """A yes/no answer from the probability of yes."""
        return cls(
            YES if probability >= YES_AT else NO,
            {YES: probability, NO: 1 - probability},
        )

    @property
    def is_valid(self) -> bool:
        """Whether every probability is from 0 to 1; a provider's may not be."""
        return all(0 <= p <= 1 for p in self.probabilities.values())

    def probability(self, label: str = YES) -> float:
        """The probability of ``label`` (of yes, by default)."""
        return self.probabilities.get(label, 0.0)


@functools.lru_cache(maxsize=4096)
def _certain(label: str, labels: tuple[str, ...]) -> Answer:
    return Answer(label, {option: float(option == label) for option in labels})
