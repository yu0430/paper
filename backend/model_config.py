"""Central model configuration and provider factories.

The application can use either Alibaba Cloud Model Studio (Qwen) or a
locally running Ollama instance for chat. Embeddings currently use Qwen's
OpenAI-compatible embedding endpoint.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


load_dotenv()


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _embedding_dim() -> int:
    raw_value = os.getenv("EMBEDDING_DIM", "1024").strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError("EMBEDDING_DIM must be an integer.") from exc
    if value <= 0:
        raise RuntimeError("EMBEDDING_DIM must be greater than zero.")
    return value


@dataclass(frozen=True)
class ModelSettings:
    llm_provider: str
    embedding_provider: str
    dashscope_base_url: str
    qwen_chat_model: str
    qwen_embedding_model: str
    embedding_dim: int
    ollama_base_url: str
    ollama_model: str


def load_model_settings() -> ModelSettings:
    llm_provider = os.getenv("LLM_PROVIDER", "qwen").strip().lower()
    if llm_provider not in {"qwen", "ollama"}:
        raise RuntimeError("LLM_PROVIDER must be either 'qwen' or 'ollama'.")

    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "qwen").strip().lower()
    if embedding_provider != "qwen":
        raise RuntimeError("EMBEDDING_PROVIDER currently supports only 'qwen'.")

    return ModelSettings(
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
        dashscope_base_url=os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ).strip().rstrip("/"),
        qwen_chat_model=os.getenv("QWEN_CHAT_MODEL", "qwen3.6-flash").strip(),
        qwen_embedding_model=os.getenv(
            "QWEN_EMBEDDING_MODEL", "text-embedding-v4"
        ).strip(),
        embedding_dim=_embedding_dim(),
        ollama_base_url=os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        ).strip().rstrip("/"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2:3b").strip(),
    )


settings = load_model_settings()


def create_chat_model(temperature: float = 0.1):
    """Create the configured chat model without leaking provider details."""
    if settings.llm_provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise RuntimeError(
                "LLM_PROVIDER is 'ollama', but langchain-ollama is not installed."
            ) from exc
        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
        )

    return ChatOpenAI(
        model=settings.qwen_chat_model,
        api_key=_required_env("DASHSCOPE_API_KEY"),
        base_url=settings.dashscope_base_url,
        temperature=temperature,
        extra_body={"enable_thinking": False},
    )


def create_embedding_model() -> OpenAIEmbeddings:
    """Create Qwen embeddings through Model Studio's OpenAI-compatible API."""
    return OpenAIEmbeddings(
        model=settings.qwen_embedding_model,
        api_key=_required_env("DASHSCOPE_API_KEY"),
        base_url=settings.dashscope_base_url,
        dimensions=settings.embedding_dim,
        chunk_size=10,
        check_embedding_ctx_length=False,
    )
