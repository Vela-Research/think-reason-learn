from .ask import OpenAILLM


def get_openai_llm(api_key: str) -> OpenAILLM | None:
    return OpenAILLM(api_key) if api_key else None
