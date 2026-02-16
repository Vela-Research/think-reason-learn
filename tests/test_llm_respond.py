"""Test the LLM respond function."""

import pytest
from unittest.mock import MagicMock
from think_reason_learn.core.llms import LLM
from think_reason_learn.core.llms._schemas import OpenAIChoice


def test_error_message_interpolates_parameters():
    """Regression test for issue #37: error messages must interpolate values."""
    # Create LLM instance and mock all provider respond_sync to return None
    llm = LLM()

    # Save original methods
    orig_openai = llm.openai_llm
    orig_google = llm.google_llm
    orig_anthropic = llm.anthropic_llm
    orig_xai = llm.xai_llm

    try:
        # Mock all providers to return None (simulating failures)
        llm.openai_llm = MagicMock(respond_sync=MagicMock(return_value=None))
        llm.google_llm = MagicMock(respond_sync=MagicMock(return_value=None))
        llm.anthropic_llm = MagicMock(respond_sync=MagicMock(return_value=None))
        llm.xai_llm = MagicMock(respond_sync=MagicMock(return_value=None))

        test_query = "What is the capital of France?"
        test_instructions = "Answer concisely"
        test_temperature = 0.7
        test_kwargs = {"max_tokens": 100}

        with pytest.raises(ValueError) as exc_info:
            llm.respond_sync(
                query=test_query,
                llm_priority=[OpenAIChoice(model="gpt-4")],
                response_format=str,
                instructions=test_instructions,
                temperature=test_temperature,
                **test_kwargs,  # type: ignore[arg-type]
            )

        error_message = str(exc_info.value)

        # Assert actual values are interpolated (NOT literal template strings)
        assert (
            test_query in error_message
        ), "Query should be interpolated"
        assert (
            test_instructions in error_message
        ), "Instructions should be interpolated"
        assert (
            "0.7" in error_message
        ), "Temperature should be interpolated"
        assert (
            "100" in error_message or "max_tokens" in error_message
        ), "kwargs should be interpolated"

        # Assert literal template strings are NOT present
        assert (
            "{query}" not in error_message
        ), "Should not contain literal {query}"
        assert (
            "{instructions}" not in error_message
        ), "Should not contain literal {instructions}"
        assert (
            "{temperature}" not in error_message
        ), "Should not contain literal {temperature}"
        assert (
            "{kwargs}" not in error_message
        ), "Should not contain literal {kwargs}"

    finally:
        # Restore original methods
        llm.openai_llm = orig_openai
        llm.google_llm = orig_google
        llm.anthropic_llm = orig_anthropic
        llm.xai_llm = orig_xai

