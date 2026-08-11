from datetime import date
from pydantic import BaseModel, ConfigDict, Field
from llm_foundations.types import Usage


class ModelPricing(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_per_mtok: float = Field(..., ge=0)
    output_per_mtok: float = Field(..., ge=0)
    currency: str = "USD"
    last_verified: date


class Cost(BaseModel):
    model_config = ConfigDict(frozen=True)

    billed_usd: float = Field(..., ge=0)
    shadow_usd: float = Field(..., ge=0)
    currency: str = "USD"

    @property
    def is_free(self) -> bool:
        return self.billed_usd == 0.0 and self.shadow_usd > 0.0


PRICING: dict[str, ModelPricing] = {
    "gemini-3.5-flash": ModelPricing(
        input_per_mtok=1.50,
        output_per_mtok=9.00,
        last_verified=date(2026, 8, 11),
    ),
}


def cost_for(usage: Usage, model: str, *, free_tier: bool) -> Cost | None:
    """Price a call. Returns None if the model isn't in the table —
    callers must distinguish 'free' from 'unknown'."""
    pricing = PRICING.get(model)
    if pricing is None:
        return None

    shadow = (
        usage.input_tokens / 1_000_000 * pricing.input_per_mtok
        + usage.output_tokens / 1_000_000 * pricing.output_per_mtok
    )
    return Cost(
        billed_usd=0.0 if free_tier else shadow,
        shadow_usd=shadow,
        currency=pricing.currency,
    )