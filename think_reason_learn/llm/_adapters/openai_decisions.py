"""OpenAI's Decisions API: probabilities from GPT-6 Luna."""

from __future__ import annotations

from collections.abc import Mapping

import openai
from openai.types import Decision, decision_create_params
from openai.types.decision import Answer as DecisionAnswer

from .._config import get_config
from .._questions import Answer, ChoiceQuestion, Question
from .._spec import Endpoint
from .._usage import TokenCounts
from ._base import Answers, DecisionAdapter, Reply, classify
from .openai_compat import openai_client


class OpenAIDecisionsAdapter(DecisionAdapter):
    """``client.decisions.create`` over the OpenAI SDK."""

    async def answer(
        self,
        endpoint: Endpoint,
        sample: str,
        questions: Mapping[str, Question],
        dropped: frozenset[str],
    ) -> Reply[Answers]:
        """Answers with GPT-6 Luna's probabilities."""
        client = openai_client(endpoint)
        try:
            raw = await client.decisions.with_raw_response.create(
                model=endpoint.model_id,
                input=sample,
                questions=[_wire(qid, q) for qid, q in questions.items()],
                timeout=get_config().timeout,
            )
        except openai.APIStatusError as exc:
            raise classify(exc.status_code, exc.message, exc.response.headers) from exc
        except openai.APIConnectionError as exc:
            raise classify(None, str(exc), {}) from exc
        decision = raw.parse()
        by_name = {a.name: a for a in decision.answers if a.name is not None}
        return Reply(
            {qid: _answer(q, by_name.get(qid)) for qid, q in questions.items()},
            _tokens(decision),
            decision.model,
            headers=dict(raw.headers),
        )


def _wire(qid: str, question: Question) -> decision_create_params.Question:
    if isinstance(question, ChoiceQuestion):
        return {
            "type": "choice",
            "name": qid,
            "instructions": question.text,
            "choices": [{"value": label} for label in question.labels],
        }
    return {"type": "predicate", "name": qid, "instructions": question.text}


def _answer(question: Question, answer: DecisionAnswer | None) -> Answer | None:
    """The answer, if it is of the question's kind and names one of its labels."""
    if answer is None:
        return None
    if answer.type == "predicate":
        if isinstance(question, ChoiceQuestion):
            return None
        return Answer.yes_probability(answer.probability)
    if answer.type != "choice" or str(answer.choice) not in question.labels:
        return None
    label = str(answer.choice)
    probabilities = {str(p.value): p.probability for p in answer.probabilities}
    if not probabilities:
        return Answer.certain(label, question.labels)
    return Answer(label, probabilities)


def _tokens(decision: Decision) -> TokenCounts:
    usage = decision.usage
    return TokenCounts(
        input=usage.input_tokens,
        output=usage.output_tokens,
        cache_read=usage.input_tokens_details.cached_tokens,
        cache_write=usage.input_tokens_details.cache_write_tokens,
        reasoning=usage.output_tokens_details.reasoning_tokens,
    )
