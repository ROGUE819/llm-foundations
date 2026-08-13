from enum import StrEnum
from typing import Literal
from pydantic import BaseModel, Field, computed_field, field_validator, ConfigDict

class Provider(StrEnum):
    GEMINI = "gemini"
    GROQ = "groq"
    OPENROUTER = "openrouter"

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1)

    def to_api(self) -> dict[str, str]:
        return {
            "role": self.role,
            "content": self.content,
        }

class Usage(BaseModel):
    input_tokens: int = Field(..., ge=0)
    output_tokens: int = Field(..., ge=0)

    @computed_field
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class RunResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    content: str = ""
    usage: Usage | None = None
    model: str
    provider: Provider
    ttft_ms: float | None = None
    latency_ms: float = Field(..., ge=0)
    finish_reason: str | None = None

    @field_validator("content", mode="after")
    @classmethod
    def strip_content(cls, content: str) -> str:
        return content.strip()

    @property
    def was_truncated(self) -> bool:
        return self.finish_reason == "length"

class TextDelta(BaseModel):
    model_config = ConfigDict(frozen=True)
    text: str

StreamEvent = TextDelta | RunResult