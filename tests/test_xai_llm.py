"""Test xAI LLM provider instruction injection.

Regression tests for issue #36: generator object truthiness bug
that caused system instructions to be silently dropped.
"""

from unittest.mock import MagicMock, patch


# Custom dict that excludes 'messages' key when unpacked with **
# but includes it for .get()
class KwargsDict(dict):
    """Dict that excludes 'messages' during ** unpacking."""

    def keys(self):  # type: ignore[override]
        """Return keys excluding 'messages'."""
        return [k for k in super().keys() if k != "messages"]

    def items(self):  # type: ignore[override]
        """Return items excluding 'messages'."""
        return [(k, v) for k, v in super().items() if k != "messages"]

    def __iter__(self):
        """Iterate over keys excluding 'messages'."""
        return iter([k for k in super().__iter__() if k != "messages"])


# Mock xAI SDK types (for offline testing)
class MockMessageRole:
    """Mock xAI SDK MessageRole enum."""

    ROLE_SYSTEM = 1
    ROLE_USER = 2
    ROLE_ASSISTANT = 3


class MockMessage:
    """Mock xAI SDK Message class."""

    def __init__(self, role, content=""):
        """Initialize mock message with role and content."""
        self.role = role
        self.content = content


def test_instructions_injected_when_no_system_message_sync():
    """Regression for #36: instructions injected when no system message (sync)."""
    with patch("think_reason_learn.core.llms._xai.ask.Client") as MockClient:
        with patch("think_reason_learn.core.llms._xai.ask.system") as mock_system:
            with patch(
                "think_reason_learn.core.llms._xai.ask.MessageRole", MockMessageRole
            ):
                # Setup mocks
                mock_client_instance = MagicMock()
                MockClient.return_value = mock_client_instance

                # Mock system() to return a mock Message with ROLE_SYSTEM
                system_msg = MockMessage(
                    role=MockMessageRole.ROLE_SYSTEM, content="test instructions"
                )
                mock_system.return_value = system_msg

                # Mock chat.create to capture calls
                # Use *args, **kwargs to avoid duplicate kwarg errors
                captured_calls = []

                def mock_create(*args, **kwargs):
                    # Capture the messages that were passed
                    captured_calls.append({"messages": kwargs.get("messages")})
                    # Return mock response with .sample() method
                    mock_sample_result = MagicMock()
                    mock_sample_result.content = "test response"
                    mock_sample_result.logprobs.content = [
                        MagicMock(token="test", logprob=-0.1)
                    ]
                    mock_sample_result.usage.total_tokens = 10

                    mock_response = MagicMock()
                    mock_response.sample.return_value = mock_sample_result
                    return mock_response

                mock_client_instance.chat.create = mock_create

                # Create xAI LLM instance
                from think_reason_learn.core.llms._xai.ask import xAILLM

                llm = xAILLM(api_key="fake-key")

                # Patch _process_kwargs to return KwargsDict
                # Allows .get("messages") but excludes during ** unpacking
                def patched_process(kwargs):
                    return KwargsDict(kwargs)

                llm._process_kwargs = patched_process

                # Call respond_sync with instructions but no system message in messages
                test_instructions = "Be concise and helpful"
                user_msg = MockMessage(
                    role=MockMessageRole.ROLE_USER, content="What is 2+2?"
                )

                llm.respond_sync(
                    query="What is 2+2?",
                    model="grok-beta",
                    response_format=str,
                    instructions=test_instructions,
                    messages=[user_msg],
                )

                # Assert chat.create was called
                assert len(captured_calls) == 1

                # Get the messages argument passed to chat.create
                messages_sent = captured_calls[0]["messages"]

                # Assert: system message was injected at position 0
                assert len(messages_sent) >= 2, "Should have system + user messages"
                assert (
                    messages_sent[0].role == MockMessageRole.ROLE_SYSTEM
                ), "First message should be system message"
                assert (
                    messages_sent[0] == system_msg
                ), "System message should contain instructions"

                # Verify system() was called with correct instructions
                mock_system.assert_called_with(test_instructions)
