"""Prompt presets for domain-specific RRF pipelines."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptPreset:
    """Named prompt collection for a specific RRF domain.

    A preset provides domain-specific system messages and user templates that
    bypass the default meta-prompt step, giving tighter control over question
    generation and answering behaviour.

    Attributes:
        name: Short identifier (used for registry lookup).
        description: Human-readable description of the preset.
        question_gen_system: System message for the question-generation LLM call.
        question_gen_user_template: User prompt template for question generation.
            Must contain ``{num_questions}`` and ``{samples}`` placeholders.
        question_answer_system: System message for the question-answering LLM call.
        question_answer_user_template: User prompt template for question answering.
            Must contain ``{question}`` and ``{sample}`` placeholders.
    """

    name: str
    description: str
    question_gen_system: str
    question_gen_user_template: str
    question_answer_system: str
    question_answer_user_template: str


VC_FOUNDER_PRESET = PromptPreset(
    name="vc_founder_evaluation",
    description=(
        "Original RRF submission prompts for VC founder evaluation "
        "(anonymised summaries)."
    ),
    question_gen_system=(
        "You are a VC analyst specialising in evaluating founders. Your task "
        "is to generate high-quality, objective YES/NO questions based on "
        "provided founder summaries. These questions should evaluate factors "
        "such as education, work history, leadership, and relevant experience."
    ),
    question_gen_user_template=(
        "Generate {num_questions} YES/NO questions that can be answered using "
        "the anonymised founder summaries below.\n"
        "Each question should be objective and specific, relating to "
        "observable traits in the summaries (e.g., education, roles, "
        "founding experience).\n"
        "IMPORTANT FORMATTING REQUIREMENTS:\n"
        "1. Return ONLY the questions, one per line\n"
        "2. Do NOT include any introductory text or explanations\n"
        "3. Do NOT include any numbering or bullet points\n"
        "4. Each line should contain exactly one question that can be "
        "answered with Yes or No\n\n"
        "Example format (DO NOT USE THIS QUESTION):\n"
        "Has the founder previously held a senior leadership role?\n\n"
        "Founder Summaries:\n\n{samples}"
    ),
    question_answer_system=(
        "You are a VC analyst evaluating founders. Your task is to decide "
        "whether the provided question applies to each founder based on their "
        "anonymised summary. Return your answer as 'Yes' or 'No'."
    ),
    question_answer_user_template=(
        "Given the following anonymised founder summary, answer the question "
        "concisely as 'Yes' or 'No'.\n\n"
        "**Question:** {question}\n\n"
        "**Sample:**\n{sample}"
    ),
)

PROMPT_PRESETS: dict[str, PromptPreset] = {
    VC_FOUNDER_PRESET.name: VC_FOUNDER_PRESET,
}
