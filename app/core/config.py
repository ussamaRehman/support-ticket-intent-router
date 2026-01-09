from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    MODEL_DIR: Optional[str] = None
    DEFAULT_TOP_K: int = 3
    MAX_TEXT_CHARS: int = 2000
    MAX_BODY_BYTES: int = 262144
    PREDICT_TIMEOUT_MS: int = 1500


def get_settings() -> Settings:
    return Settings()
