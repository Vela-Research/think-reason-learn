from .ask import GeminiLLM


def get_google_llm(api_key: str) -> GeminiLLM | None:
    return GeminiLLM(api_key) if api_key else None
