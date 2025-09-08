from .ask import xAILLM


def get_xai_llm(api_key: str) -> xAILLM | None:
    return xAILLM(api_key) if api_key else None
