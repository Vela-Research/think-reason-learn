from .ask import AnthropicLLM


def get_anthropic_llm(api_key: str) -> AnthropicLLM | None:
    return AnthropicLLM(api_key) if api_key else None
