from datetime import UTC, datetime
from statistics import quantiles

from pydantic import BaseModel, ConfigDict, Field

from llm_foundations.pricing import Cost
from llm_foundations.types import RunResult, Usage


class LedgerEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    usage: Usage
    billed_usd: float = Field(..., ge=0)
    shadow_usd: float = Field(..., ge=0)
    latency_ms: float = Field(..., ge=0)
    timestamp: datetime


class SessionLedger(BaseModel):
    entries: list[LedgerEntry] = Field(default_factory=list)
    unpriced_calls: int = 0
    unmeasured_calls: int = 0

    def record(self, result: RunResult, cost: Cost | None) -> None:
        if result.usage is None:
            self.unmeasured_calls += 1
        if cost is None:
            self.unpriced_calls += 1

        self.entries.append(
            LedgerEntry(
                model=result.model,
                usage=result.usage or Usage(input_tokens=0, output_tokens=0),
                billed_usd=cost.billed_usd if cost else 0.0,
                shadow_usd=cost.shadow_usd if cost else 0.0,
                latency_ms=result.latency_ms,
                timestamp=datetime.now(UTC),
            )
        )

    @property
    def call_count(self) -> int:
        return len(self.entries)

    @property
    def total_input_tokens(self) -> int:
        return sum(e.usage.input_tokens for e in self.entries)

    @property
    def total_output_tokens(self) -> int:
        return sum(e.usage.output_tokens for e in self.entries)

    @property
    def total_billed(self) -> float:
        return sum(e.billed_usd for e in self.entries)

    @property
    def total_shadow(self) -> float:
        return sum(e.shadow_usd for e in self.entries)

    @property
    def mean_latency_ms(self) -> float:
        if not self.entries:
            return 0.0
        return sum(e.latency_ms for e in self.entries) / len(self.entries)

    @property
    def p95_latency_ms(self) -> float:
        if not self.entries:
            return 0.0
        latencies = sorted(e.latency_ms for e in self.entries)
        if len(latencies) < 2:
            return latencies[0]
        return float(quantiles(latencies, n=100)[94])

    def summary(self) -> str:
        parts = [
            f"{self.call_count} calls",
            f"{self.total_input_tokens:,} in / {self.total_output_tokens:,} out",
            f"${self.total_shadow:.4f} shadow",
            f"p95 {self.p95_latency_ms:.0f}ms",
        ]
        caveats = []
        if self.unpriced_calls:
            caveats.append(f"{self.unpriced_calls} unpriced")
        if self.unmeasured_calls:
            caveats.append(f"{self.unmeasured_calls} without usage")
        if caveats:
            parts.append(f"({', '.join(caveats)})")
        return " · ".join(parts)