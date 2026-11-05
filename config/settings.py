from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Investor Intelligence Platform"
    app_version: str = "0.1.0"
    app_env: Literal["development", "testing", "production"] = "development"
    debug: bool = False

    # Azure OpenAI
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: SecretStr | None = None
    azure_openai_embedding_deployment: str = "text-embedding-3-small"

    # Azure AI Search
    azure_search_endpoint: str | None = None
    azure_search_api_key: SecretStr | None = None
    azure_search_index_name: str = "financial-document-chunks"

    # PostgreSQL
    db_host: str = "127.0.0.1"
    db_port: int = 5432
    db_name: str = "investor_intelligence"
    db_user: str = "investor_app"
    db_password: SecretStr | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
