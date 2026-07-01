import os
from pathlib import Path


class Settings:
    """Central configuration loaded from environment variables.

    All LLM providers that expose an OpenAI-compatible /chat/completions
    endpoint work here (OpenAI, Groq, OpenRouter, local vLLM, etc.) --
    just change LLM_BASE_URL / LLM_API_KEY / LLM_MODEL.
    """

    def __init__(self) -> None:
        self.llm_api_key: str = os.environ.get("LLM_API_KEY", "")
        self.llm_base_url: str = os.environ.get(
            "LLM_BASE_URL", "https://api.groq.com/openai/v1"
        )
        self.llm_model: str = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
        self.llm_temperature_extract: float = float(
            os.environ.get("LLM_TEMPERATURE_EXTRACT", "0.0")
        )
        self.llm_temperature_generate: float = float(
            os.environ.get("LLM_TEMPERATURE_GENERATE", "0.3")
        )
        self.llm_timeout_seconds: float = float(
            os.environ.get("LLM_TIMEOUT_SECONDS", "20")
        )
        self.llm_max_retries: int = int(os.environ.get("LLM_MAX_RETRIES", "2"))

        base_dir = Path(__file__).resolve().parent.parent
        self.catalog_path: str = os.environ.get(
            "CATALOG_PATH", str(base_dir / "data" / "shl_catalog.json")
        )

        self.max_turns: int = int(os.environ.get("MAX_TURNS", "8"))
        self.max_recommendations: int = int(os.environ.get("MAX_RECOMMENDATIONS", "10"))
        self.retrieval_pool_size: int = int(os.environ.get("RETRIEVAL_POOL_SIZE", "30"))
        self.request_timeout_seconds: float = float(
            os.environ.get("REQUEST_TIMEOUT_SECONDS", "28")
        )


settings = Settings()
