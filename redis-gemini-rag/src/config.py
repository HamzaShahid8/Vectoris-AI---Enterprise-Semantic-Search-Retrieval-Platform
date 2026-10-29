from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    redis_host: str
    redis_port: int
    redis_db: int
    redis_index_name: str
    redis_doc_prefix: str
    data_dir: Path
    gemini_embedding_model: str
    gemini_chat_model: str
    embedding_dim: int
    chunk_size: int
    chunk_overlap: int
    top_k: int
    gemini_debug: bool
    llm_provider: str
    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_chat_model: str
    embedding_provider: str


def _get_required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        if name == "GEMINI_API_KEY":
            raise ValueError("Missing GEMINI_API_KEY. Please add it to your .env file.")
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def load_settings() -> Settings:
    load_dotenv()

    data_dir = Path(_get_required_env("DATA_DIR")).resolve()
    if not data_dir.exists() or not data_dir.is_dir():
        raise ValueError(f"Data folder does not exist: {data_dir}")

    llm_provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if llm_provider not in {"gemini", "deepseek"}:
        raise ValueError("LLM_PROVIDER must be either 'gemini' or 'deepseek'.")

    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "gemini").strip().lower()
    if embedding_provider not in {"gemini", "local"}:
        raise ValueError("EMBEDDING_PROVIDER must be either 'gemini' or 'local'.")

    gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if llm_provider == "gemini" or embedding_provider == "gemini":
        if not gemini_api_key:
            raise ValueError("Missing GEMINI_API_KEY. Please add it to your .env file.")

    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    deepseek_chat_model = os.getenv("DEEPSEEK_CHAT_MODEL", "").strip()
    if llm_provider == "deepseek":
        if not deepseek_api_key:
            raise ValueError("Missing DEEPSEEK_API_KEY. Please add it to your .env file.")
        if not deepseek_chat_model:
            raise ValueError("Missing DEEPSEEK_CHAT_MODEL. Please add it to your .env file.")

    gemini_embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "").strip()
    gemini_chat_model = os.getenv("GEMINI_CHAT_MODEL", "").strip()
    if embedding_provider == "gemini" and not gemini_embedding_model:
        raise ValueError("Missing GEMINI_EMBEDDING_MODEL. Please add it to your .env file.")
    if llm_provider == "gemini" and not gemini_chat_model:
        raise ValueError("Missing GEMINI_CHAT_MODEL. Please add it to your .env file.")

    return Settings(
        gemini_api_key=gemini_api_key,
        redis_host=os.getenv("REDIS_HOST", "localhost"),
        redis_port=int(os.getenv("REDIS_PORT", "6379")),
        redis_db=int(os.getenv("REDIS_DB", "0")),
        redis_index_name=os.getenv("REDIS_INDEX_NAME", "rag_index"),
        redis_doc_prefix=os.getenv("REDIS_DOC_PREFIX", "rag:chunk:"),
        data_dir=data_dir,
        gemini_embedding_model=gemini_embedding_model or "gemini-embedding-001",
        gemini_chat_model=gemini_chat_model or "gemini-2.0-flash",
        embedding_dim=int(_get_required_env("EMBEDDING_DIM")),
        chunk_size=int(_get_required_env("CHUNK_SIZE")),
        chunk_overlap=int(_get_required_env("CHUNK_OVERLAP")),
        top_k=int(_get_required_env("TOP_K")),
        gemini_debug=os.getenv("GEMINI_DEBUG", "false").strip().lower() == "true",
        llm_provider=llm_provider,
        deepseek_api_key=deepseek_api_key,
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip(),
        deepseek_chat_model=deepseek_chat_model,
        embedding_provider=embedding_provider,
    )
