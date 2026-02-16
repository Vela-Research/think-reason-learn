"""Fake LLM for testing RRF without making actual API calls."""

from typing import Any, List, Dict
from think_reason_learn.core.llms import LLMResponse, LLMChoice, OpenAIChoice
from think_reason_learn.rrf._rrf import Questions, Answer


class FakeLLM:
    """Drop-in LLM replacement for testing.

    Tracks all calls and returns canned responses based on response_format.
    Supports 4 RRF call patterns:
    - set_tasks → str (template string)
    - _generate_questions → Questions (Pydantic model)
    - _answer_single_question → Answer (Pydantic model)
    - _answer_questions_batch → str (JSON list)
    """

    def __init__(
        self,
        default_answer: str = "YES",
        questions_per_call: int = 3,
        template_str: str = "Generate <number_of_questions> YES/NO questions",
    ):
        """Initialize FakeLLM.

        Args:
            default_answer: Default answer for questions ("YES" or "NO")
            questions_per_call: Number of questions to return per generation call
            template_str: Template string to return for set_tasks
                (must contain <number_of_questions>)
        """
        self.call_count = 0
        self.calls: List[Dict[str, Any]] = []
        self.default_answer = default_answer
        self.questions_per_call = questions_per_call
        self.template_str = template_str

    async def respond(
        self,
        query: str,
        llm_priority: LLMChoice | None = None,
        response_format: type | None = None,
        instructions: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse[Any]:
        """Async respond method matching LLM.respond() signature.

        Args:
            query: The query string
            llm_priority: LLM choice (ignored)
            response_format: Expected response type (str, Questions, Answer)
            instructions: Instructions for the LLM (ignored)
            **kwargs: Additional arguments (ignored)

        Returns:
            LLMResponse with appropriate response type
        """
        self.call_count += 1
        self.calls.append(
            {
                "query": query,
                "llm_priority": llm_priority,
                "response_format": response_format,
                "instructions": instructions,
                "kwargs": kwargs,
            }
        )

        # Determine response based on response_format
        if response_format == Questions:
            # Generate questions
            response = Questions(
                questions=[f"Question {i + 1}" for i in range(self.questions_per_call)],
                cumulative_memory="Test memory",
            )
        elif response_format == Answer:
            # Answer single question
            response = Answer(answer=self.default_answer)  # type: ignore
        elif response_format is str:
            # Check if this is a batch answer call or template call
            if "SAMPLES:" in query and "[" in query:
                # Batch answering - return JSON list
                # Count how many samples by counting newlines between sample markers
                num_samples = query.count("SAMPLE ")
                response = (
                    "["
                    + ", ".join(
                        [f'"{self.default_answer}"' for _ in range(num_samples)]
                    )
                    + "]"
                )
            else:
                # Template generation
                response = self.template_str  # type: ignore
        else:
            # Default to empty string
            response = ""  # type: ignore

        return LLMResponse(
            response=response,
            logprobs=[("token", -0.1)],
            total_tokens=10,
            provider_model=OpenAIChoice(provider="openai", model="gpt-4o-mini"),
        )

    def respond_sync(
        self,
        query: str,
        llm_priority: LLMChoice | None = None,
        response_format: type | None = None,
        instructions: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse[Any]:
        """Sync version of respond (not used by RRF, but provided for completeness)."""
        # Increment counter synchronously
        self.call_count += 1
        self.calls.append(
            {
                "query": query,
                "llm_priority": llm_priority,
                "response_format": response_format,
                "instructions": instructions,
                "kwargs": kwargs,
            }
        )

        # Same logic as async version
        if response_format == Questions:
            response = Questions(
                questions=[f"Question {i + 1}" for i in range(self.questions_per_call)],
                cumulative_memory="Test memory",
            )
        elif response_format == Answer:
            response = Answer(answer=self.default_answer)  # type: ignore
        elif response_format is str:
            if "SAMPLES:" in query and "[" in query:
                num_samples = query.count("SAMPLE ")
                response = (
                    "["
                    + ", ".join(
                        [f'"{self.default_answer}"' for _ in range(num_samples)]
                    )
                    + "]"
                )
            else:
                response = self.template_str  # type: ignore
        else:
            response = ""  # type: ignore

        return LLMResponse(
            response=response,
            logprobs=[("token", -0.1)],
            total_tokens=10,
            provider_model=OpenAIChoice(provider="openai", model="gpt-4o-mini"),
        )

    def reset(self) -> None:
        """Reset call tracking."""
        self.call_count = 0
        self.calls = []
