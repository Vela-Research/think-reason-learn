from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Literal, Sequence, Tuple, Type, cast

from pydantic import BaseModel, ValidationError, create_model

from think_reason_learn.core._config import settings
from think_reason_learn.core._singleton import SingletonMeta

from ._anthropic.ask import AnthropicLLM, get_anthropic_llm
from ._anthropic.schemas import AnthropicChoice
from ._google.ask import GeminiLLM, get_google_llm
from ._google.schemas import GoogleChoice
from ._jev.client import (
    JevBudget,
    JevClient,
    JevRequest,
    JevResult,
    get_jev_client,
)
from ._jev.schemas import AnswerValue, ChoiceQuestion, JevChoice, JevQuestion
from ._openai.ask import OpenAILLM, get_openai_llm
from ._openai.schemas import OpenAIChoice
from ._schemas import (
    NOT_GIVEN,
    LLMChoice,
    LLMChoiceModel,
    LLMResponse,
    NotGiven,
    T,
    TokenCounter,
)
from ._xai.ask import get_xai_llm, xAILLM
from ._xai.schemas import XAIChoice

logger = logging.getLogger(__name__)

JEV_CANNOT_GENERATE = (
    "Jev answers questions about samples and cannot generate text. Use a chat "
    "model (OpenAI, Google, Anthropic or xAI) for generation, critique and "
    "instructions, and Jev for answering."
)

ANSWER_INSTRUCTIONS = (
    "You answer one question about one sample. Base the answer only on the "
    "sample's content and reply in the requested format."
)


class _YesNo(BaseModel):
    answer: Literal["YES", "NO"]


def _provider(choice: LLMChoice) -> str:
    return choice["provider"] if isinstance(choice, dict) else choice.provider


def _chat_query(state: str, question: JevQuestion) -> str:
    if isinstance(question, ChoiceQuestion):
        ask = "Answer with exactly one of: " + ", ".join(question.labels) + "."
    else:
        ask = "Answer YES or NO."
    return f"Question: {question.instructions}\n\nSample:\n{state}\n\n{ask}"


def _chat_format(question: JevQuestion) -> Type[BaseModel]:
    if isinstance(question, ChoiceQuestion):
        labels = cast(Any, Literal[tuple(question.labels)])  # type: ignore[valid-type]
        return create_model("ChoiceAnswer", answer=(labels, ...))
    return _YesNo


class LLM(metaclass=SingletonMeta):
    """A singleton class that provides a unified interface for the LLMs."""

    def __init__(self) -> None:
        self.anthropic_llm = get_anthropic_llm(settings.ANTHROPIC_API_KEY)
        self.google_llm = get_google_llm(settings.GOOGLE_AI_API_KEY)
        self.openai_llm = get_openai_llm(settings.OPENAI_API_KEY)
        self.xai_llm = get_xai_llm(settings.XAI_API_KEY)
        self.jev_llm = get_jev_client(settings.TYPESAFE_API_KEY)

    def _val_llm_priority_and_api_keys(
        self, llm_priority: List[LLMChoice]
    ) -> List[LLMChoiceModel]:
        """Validate the LLMPriority and ensure the API key is set for such provider."""
        models_map = {
            "anthropic": AnthropicChoice,
            "google": GoogleChoice,
            "openai": OpenAIChoice,
            "xai": XAIChoice,
            "jev": JevChoice,
        }
        priority_models: List[LLMChoiceModel] = []
        for llmp in llm_priority:
            if isinstance(llmp, dict):
                priority_class = models_map.get(llmp["provider"])
                if priority_class is None:
                    raise ValueError(
                        f"Invalid LLMPriority: {llmp['provider']}. "
                        f"Supported providers are {list(models_map.keys())}."
                    )

                try:
                    llmp = priority_class(**llmp)  # type: ignore
                except ValidationError:
                    supported_models = priority_class.__annotations__["model"].__args__
                    raise ValueError(
                        f"Invalid LLMPriority: {llmp['model']} for {llmp['provider']}. "
                        f"Supported models are {supported_models}."
                    )

            else:
                llmp = llmp

            if getattr(self, f"{llmp.provider}_llm") is None:
                key_name = (
                    "TYPESAFE_API_KEY"
                    if llmp.provider == "jev"
                    else f"{llmp.provider.upper()}_API_KEY"
                )
                raise ValueError(
                    f"Can't use {llmp.model}. {key_name} not set! "
                    "Explicitly set it in your environment."
                )
            priority_models.append(llmp)
        return priority_models

    def respond_sync(
        self,
        llm_priority: List[LLMChoice],
        query: str = "",
        response_format: Type[T] = str,
        instructions: str | NotGiven | None = NOT_GIVEN,
        temperature: float | NotGiven | None = NOT_GIVEN,
        **kwargs: Dict[str, Any],
    ) -> LLMResponse[T]:
        """Respond to a query using the LLM synchronously.

        Args:
            query: The query to respond to.
            llm_priority: LLMs to use in order of priority.
            response_format: The response format to use.
            instructions: Optional instructions to use.
            temperature: Optional temperature to use.
            **kwargs: Additional arguments to pass to the LLM. For:

                - OpenAI: ``openai.OpenAI.responses.parse`` or
                    ``openai.OpenAI.responses.create``.
                - Google: ``google.genai.types.GenerateContentConfig``.
                - XAI: ``xai_sdk.Client.chat.create``.
                - Anthropic: ``anthropic.Client.messages.create``.

        Note:
            Provided kwargs override the function arguments.

        Returns:
            The response from the LLM.

        Raises:
            AssertionError: If llm_priority is an empty list.
            ValueError: If none of the LLMs worked.
        """
        assert len(llm_priority) > 0, "llm_priority must be a non-empty list"
        if any(_provider(c) == "jev" for c in llm_priority):
            raise ValueError(JEV_CANNOT_GENERATE)
        raise_ = len(llm_priority) == 1

        llm_priority_models = self._val_llm_priority_and_api_keys(llm_priority)

        for idx, llmp in enumerate(llm_priority_models, 1):
            if idx > 1:
                logger.warning(f"Falling back to {llmp.model} by {llmp.provider}...")

            if llmp.provider == "google":
                google_llm = cast(GeminiLLM, self.google_llm)
                response = google_llm.respond_sync(
                    query=query,
                    model=llmp.model,
                    response_format=response_format,
                    instructions=instructions or None,
                    temperature=temperature or None,
                    raise_=raise_,
                    **kwargs,
                )
                if response is not None:
                    return response

            if llmp.provider == "openai":
                openai_llm = cast(OpenAILLM, self.openai_llm)
                response = openai_llm.respond_sync(
                    query=query,
                    model=llmp.model,
                    response_format=response_format,
                    instructions=(
                        instructions
                        if not isinstance(instructions, NotGiven)
                        else NOT_GIVEN.OPENAI_NOT_GIVEN
                    ),
                    temperature=(
                        temperature
                        if not isinstance(temperature, NotGiven)
                        else NOT_GIVEN.OPENAI_NOT_GIVEN
                    ),
                    raise_=raise_,
                    **kwargs,
                )
                if response is not None:
                    return response

            if llmp.provider == "anthropic":
                anthropic_llm = cast(AnthropicLLM, self.anthropic_llm)
                response = anthropic_llm.respond_sync(
                    query=query,
                    model=llmp.model,
                    response_format=response_format,
                    instructions=instructions or NOT_GIVEN.ANTHROPIC_NOT_GIVEN,
                    temperature=temperature or NOT_GIVEN.ANTHROPIC_NOT_GIVEN,
                    raise_=raise_,
                    **kwargs,
                )
                if response is not None:
                    return response

            if llmp.provider == "xai":
                xai_llm = cast(xAILLM, self.xai_llm)
                response = xai_llm.respond_sync(
                    query=query,
                    model=llmp.model,
                    response_format=response_format,
                    instructions=instructions or None,
                    temperature=temperature or None,
                    raise_=raise_,
                    **kwargs,
                )
                if response is not None:
                    return response

        llmps = [f"{llmp.provider}: {llmp.model}" for llmp in llm_priority_models]
        raise ValueError(
            f"Failed to respond with any of {llmps}\n"
            f"Query: {query}\n"
            f"Instructions: {instructions}\n"
            f"Temperature: {temperature}\n"
            f"**kwargs: {kwargs}"
        )

    async def respond(
        self,
        query: str,
        llm_priority: List[LLMChoice],
        response_format: Type[T],
        instructions: str | NotGiven | None = NOT_GIVEN,
        temperature: float | NotGiven | None = NOT_GIVEN,
        **kwargs: Dict[str, Any],
    ) -> LLMResponse[T]:
        """Respond to a query using the LLM asynchronously.

        Args:
            query: The query to respond to.
            llm_priority: LLMs to use in order of priority.
            response_format: The response format to use.
            instructions: Optional instructions to use.
            temperature: Optional temperature to use.
            **kwargs: Additional arguments to pass to the LLM. For:

                - OpenAI: ``openai.OpenAI.responses.parse`` or
                    ``openai.OpenAI.responses.create``.
                - Google: ``google.genai.types.GenerateContentConfig``.
                - XAI: ``xai_sdk.Client.chat.create``.
                - Anthropic: ``anthropic.Client.messages.create``.

        Note:
            Provided kwargs override the function arguments.

        Returns:
            The response from the LLM.

        Raises:
            AssertionError: If llm_priority is an empty list.
            ValueError: If none of the LLMs worked.
        """
        assert len(llm_priority) > 0, "llm_priority must be a non-empty list"
        if any(_provider(c) == "jev" for c in llm_priority):
            raise ValueError(JEV_CANNOT_GENERATE)
        raise_ = len(llm_priority) == 1

        llm_priority_models = self._val_llm_priority_and_api_keys(llm_priority)

        for idx, llmp in enumerate(llm_priority_models, 1):
            if idx > 1:
                logger.warning(f"Falling back to {llmp.model} from {llmp.provider}...")

            if llmp.provider == "google":
                google_llm = cast(GeminiLLM, self.google_llm)
                response = await google_llm.respond(
                    query=query,
                    model=llmp.model,
                    response_format=response_format,
                    instructions=instructions or None,
                    temperature=temperature or None,
                    raise_=raise_,
                    **kwargs,
                )
                if response is not None:
                    return response

            if llmp.provider == "openai":
                openai_llm = cast(OpenAILLM, self.openai_llm)
                response = await openai_llm.respond(
                    query=query,
                    model=llmp.model,
                    response_format=response_format,
                    instructions=(
                        instructions
                        if not isinstance(instructions, NotGiven)
                        else NOT_GIVEN.OPENAI_NOT_GIVEN
                    ),
                    temperature=(
                        temperature
                        if not isinstance(temperature, NotGiven)
                        else NOT_GIVEN.OPENAI_NOT_GIVEN
                    ),
                    raise_=raise_,
                    **kwargs,
                )
                if response is not None:
                    return response

            if llmp.provider == "anthropic":
                anthropic_llm = cast(AnthropicLLM, self.anthropic_llm)
                response = await anthropic_llm.respond(
                    query=query,
                    model=llmp.model,
                    response_format=response_format,
                    instructions=instructions or NOT_GIVEN.ANTHROPIC_NOT_GIVEN,
                    temperature=temperature or NOT_GIVEN.ANTHROPIC_NOT_GIVEN,
                    raise_=raise_,
                    **kwargs,
                )
                if response is not None:
                    return response

            if llmp.provider == "xai":
                xai_llm = cast(xAILLM, self.xai_llm)
                response = await xai_llm.respond(
                    query=query,
                    model=llmp.model,
                    response_format=response_format,
                    instructions=instructions or None,
                    temperature=temperature or None,
                    raise_=raise_,
                    **kwargs,
                )
                if response is not None:
                    return response

        llmps = [f"{llmp.provider}: {llmp.model}" for llmp in llm_priority_models]
        raise ValueError(f"Failed to respond with any of {llmps}")

    async def _chat_answer(
        self,
        llmp: LLMChoiceModel,
        state: str,
        question: JevQuestion,
        temperature: float | NotGiven | None,
    ) -> Tuple[AnswerValue | None, LLMResponse[Any]]:
        """Answer one question about one sample with a chat model."""
        response = await self.respond(
            query=_chat_query(state, question),
            llm_priority=[llmp],
            response_format=_chat_format(question),
            instructions=ANSWER_INSTRUCTIONS,
            temperature=temperature,
        )
        parsed = cast(Any, response.response)
        if parsed is None:
            return None, response
        if isinstance(question, ChoiceQuestion):
            return str(parsed.answer), response
        return (1.0 if parsed.answer == "YES" else 0.0), response

    async def answer_many(
        self,
        llm_priority: List[LLMChoice],
        requests: Sequence[JevRequest],
        *,
        budget: JevBudget | None = None,
        token_counter: TokenCounter | None = None,
        caller: str = "LLM.answer_many",
        temperature: float | NotGiven | None = NOT_GIVEN,
        chat_concurrency: int = 5,
    ) -> List[JevResult]:
        """Answer typed questions about samples, one request per sample.

        Jev takes a whole request (one sample, many questions) per call. A chat
        model answers one question per call: a ``noul`` question as YES (1.0) or
        NO (0.0), a ``choice`` question as one of its labels. Questions a model
        fails to answer fall back to the next model in ``llm_priority``.

        Args:
            llm_priority: Answering models in priority order.
            requests: One request per sample. Never put labels in a request.
            budget: Cost cap shared by the run. Defaults to a new budget per call
                with the Jev choice's ``max_cost_usd``.
            token_counter: Counter to add usage to.
            caller: Name recorded in the token counter.
            temperature: Temperature for chat models.
            chat_concurrency: Chat calls in flight at once.

        Returns:
            One result per request; unanswered questions are ``None``.

        Raises:
            ValueError: A model's API key is not set.
            ~think_reason_learn.core.exceptions.JevCostCapError: The run's Jev
                cost cap would be or was passed.
            ~think_reason_learn.core.exceptions.JevAuthError: Typesafe rejected
                the key.
        """
        assert len(llm_priority) > 0, "llm_priority must be a non-empty list"
        models = self._val_llm_priority_and_api_keys(llm_priority)
        results = [
            JevResult(answers={qid: None for qid in r.questions}, cached=True)
            for r in requests
        ]

        for position, llmp in enumerate(models):
            todo = [
                i
                for i, result in enumerate(results)
                if any(v is None for v in result.answers.values())
            ]
            if not todo:
                break
            if position > 0:
                logger.warning(
                    f"Falling back to {llmp.model} from {llmp.provider} "
                    f"for {len(todo)} samples..."
                )
            sub_requests = [
                JevRequest(
                    state=requests[i].state,
                    questions={
                        qid: q
                        for qid, q in requests[i].questions.items()
                        if results[i].answers[qid] is None
                    },
                )
                for i in todo
            ]

            if isinstance(llmp, JevChoice):
                jev = cast(JevClient, self.jev_llm)
                sub_results = await jev.answer_many(
                    sub_requests,
                    model=llmp.model,
                    budget=budget or JevBudget(llmp.max_cost_usd),
                    use_cache=llmp.cache,
                )
                for i, sub in zip(todo, sub_results):
                    result = results[i]
                    result.answers.update(
                        {k: v for k, v in sub.answers.items() if v is not None}
                    )
                    result.input_tokens += sub.input_tokens
                    result.output_tokens += sub.output_tokens
                    result.served_model = sub.served_model or result.served_model
                    result.error = sub.error
                    result.cached = result.cached and sub.cached
                    if token_counter is not None and not sub.cached:
                        await token_counter.append(
                            model=llmp.model,
                            provider="jev",
                            value=sub.input_tokens + sub.output_tokens,
                            caller=caller,
                        )
                continue

            semaphore = asyncio.Semaphore(chat_concurrency)

            async def ask(i: int, qid: str, state: str, question: JevQuestion) -> None:
                async with semaphore:
                    try:
                        value, response = await self._chat_answer(
                            llmp, state, question, temperature
                        )
                    except Exception as exc:
                        results[i].error = f"{llmp.model}: {exc}"
                        logger.warning(
                            f"Error answering '{question.instructions}' with "
                            f"{llmp.model}",
                            exc_info=True,
                        )
                        return
                results[i].answers[qid] = value
                results[i].cached = False
                if token_counter is not None:
                    await token_counter.append(
                        model=response.provider_model.model,
                        provider=response.provider_model.provider,
                        value=response.total_tokens,
                        caller=caller,
                    )

            async with asyncio.TaskGroup() as tg:
                for i, sub in zip(todo, sub_requests):
                    for qid, question in sub.questions.items():
                        tg.create_task(ask(i, qid, sub.state, question))

        for result in results:
            if all(v is not None for v in result.answers.values()):
                result.error = None
        return results

    async def answer(
        self,
        llm_priority: List[LLMChoice],
        state: str,
        questions: Dict[str, JevQuestion],
        **kwargs: Any,
    ) -> Dict[str, AnswerValue | None]:
        """Answer typed questions about one sample. See ``answer_many``.

        Returns:
            ``{question_id: answer}``: a probability of yes for ``noul``
            questions, a label for ``choice`` questions, ``None`` if unanswered.
        """
        [result] = await self.answer_many(
            llm_priority, [JevRequest(state=state, questions=dict(questions))], **kwargs
        )
        return result.answers
