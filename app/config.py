"""Configuration for comboocr: vLLM URL, model, timeouts."""
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # vLLM server (OpenAI-compatible API)
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_api_key: Optional[str] = None  # optional, vLLM often has no key
    vllm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"  # model name as registered on vLLM
    vllm_timeout_seconds: float = 120.0
    vllm_max_tokens: int = 512
    # Для таблиц нужен больший лимит (рукописный текст, много ячеек)
    vllm_max_tokens_table: int = 2048


@lru_cache
def get_settings() -> Settings:
    return Settings()

