import asyncio
import time
from openai import AsyncOpenAI, APIStatusError, APIConnectionError, RateLimitError
from collections.abc import Awaitable, Callable
from typing import TypeVar
from openai.types.chat import ChatCompletion

from llm_foundations.ledger import SessionLedger
from llm_foundations.config import ProviderConfig, get_settings
from llm_foundations.types import Message, RunResult, Usage
from llm_foundations.pricing import cost_for, Cost

RETRYABLE = (RateLimitError, APIConnectionError)
T = TypeVar("T")

class LLMClient:
    def __init__(self, provider: ProviderConfig, max_tokens: int, temperature: float) -> None:
        self._provider = provider
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = AsyncOpenAI(
            api_key=provider.api_key,
            base_url=provider.base_url,
            timeout=30.0,
            max_retries=3,
        )

    async def complete(self, messages: list[Message]) -> tuple[RunResult, Cost | None]:
        async def _call():
            return await self._client.chat.completions.create(model=self._provider.model,
                                                              messages=[m.to_api() for m in messages],
                                                              max_tokens=self._max_tokens,
                                                              temperature=self._temperature,)

        start = time.perf_counter()
        resp = await self._with_retry(_call)
        latency_ms = (time.perf_counter() - start) * 1000
        content = resp.choices[0].message.content or ""
        finish_reason = resp.choices[0].finish_reason

        usage = (
            Usage(
               input_tokens=resp.usage.prompt_tokens,
               output_tokens=resp.usage.completion_tokens,
            )
            if resp.usage is not None
            else None
        )

        result = RunResult(
            content=content,
            finish_reason=finish_reason,
            latency_ms=latency_ms,
            usage=usage,
            model=self._provider.model,
            provider=self._provider.name,
        )
        cost = (cost_for(result.usage, result.model, free_tier=self._provider.is_free_tier) 
                if result.usage is not None 
                else None
                )
        return result, cost

    async def _with_retry(self, fn: Callable[[], Awaitable[T]], attempts: int = 4) -> T:        
        delay = 1.0
        for attempt in range(1, attempts + 1):
            try:
                return await fn()
            except RETRYABLE as exc:
                if attempt == attempts:
                    raise
                print(f"[retry {attempt}/{attempts}] {type(exc).__name__}, waiting {delay:.1f}s")
                await asyncio.sleep(delay)
                delay *= 2
        raise RuntimeError("unreachable: retry loop exhausted without raising or returning")

async def _smoke() -> None:
    s = get_settings()
    c = LLMClient(s.provider(), s.llm_max_tokens, s.llm_temperature)
    ledger = SessionLedger()
    for prompt in ["Hi", "Name three Himalayan passes.", "Explain acclimatization briefly"]:
        r, cost = await c.complete([Message(role="user", content=prompt)])
        ledger.record(r, cost)

        usage_str = (
            f"{r.usage.input_tokens} in / {r.usage.output_tokens} out"
            if r.usage else "!! NO USAGE"
        )
        cost_str = (
            f"${cost.billed_usd:.4f} (would be ${cost.shadow_usd:.4f})"
            if cost else "$?.???? unpriced"
        )
        print(f"{usage_str} · {cost_str} · {r.latency_ms:.0f}ms")

    print(f"\nSession: {ledger.summary()}")

if __name__ == "__main__":
    asyncio.run(_smoke())