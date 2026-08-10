from enum import StrEnum
from functools import lru_cache
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from llm_foundations.types import Provider

class ProviderConfig(BaseModel):
    name: Provider
    api_key: str = Field(min_length=10)
    base_url: str
    model: str
    is_free_tier: bool = True

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: Provider = Provider.GEMINI
    llm_max_tokens: int = Field(default=2048, ge=0, le=32_000)
    llm_temperature: float = Field(default=1.0, ge=0.0, le=2.0)

    gemini_api_key: str | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    gemini_model: str = "gemini-2.5-flash"

    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "llama-3.3-70b-versatile"

    @model_validator(mode="after")
    def selected_provider_has_key(self) -> "Settings":
        if getattr(self, f"{self.llm_provider.value}_api_key") is None:
            raise ValueError(f"LLM_PROVIDER={self.llm_provider.value} but "
                             f"{self.llm_provider.value.upper()}_API_KEY is not set in .env")
        return self

    def provider(self) -> ProviderConfig:
        p = self.llm_provider.value
        return ProviderConfig(
            name=self.llm_provider,
            api_key=getattr(self, f"{p}_api_key"),
            base_url=getattr(self, f"{p}_base_url"),
            model=getattr(self, f"{p}_model"),
        )

@lru_cache
def get_settings() -> Settings:
    return Settings()