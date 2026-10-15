"""GPTree's prompts: insights, questions per node, and the critic."""

from __future__ import annotations

from typing import Final

INSTRUCTIONS_PURPOSE: Final = (
    "questions that split a decision tree: each question is answered from one "
    "sample's text alone, and good questions separate the classes"
)

INSIGHT_INSTRUCTIONS: Final = """\
You study labelled samples for a classification task. Summarise, in a few short
bullet points, the patterns that the samples you are shown have in common and
that may tell their class apart from the others. Describe patterns, not
individual samples, and do not invent facts."""

MERGE_INSIGHT_INSTRUCTIONS: Final = """\
Merge these summaries of one class into a single short list of the most useful
patterns. Drop duplicates and anything specific to a single sample."""

QUESTION_INSTRUCTIONS: Final = """\
You write questions for one node of a decision tree. Another model will answer
each question from one sample's text alone, and the tree will split the node's
samples by the answers.

Write questions that:
- separate the classes among the samples at this node, given the path so far;
- can be answered from a sample's text alone;
- ask one thing each, clearly, without restating or asking for the label;
- differ from each other and from the questions already on the path.
{options}"""

YES_NO: Final = "- are answered YES or NO; leave options empty."

CHOICE: Final = """\
- are answered YES or NO (leave options empty), or with one of 3 to {max_choices}
  options you list, which must not overlap and must cover every likely answer."""

CRITIC_INSTRUCTIONS: Final = """\
You review candidate questions for one node of a decision tree. Another model
answers each question from one sample's text alone.

Drop or fix any question that cannot be answered from the text alone, leaks or
restates the label, is vague or asks two things, duplicates another, or has
options that overlap or miss likely answers. Return the questions that pass,
fixed where needed, and at least one. Never return more than you were given."""
