# llm-foundations

Shared core for LLM application work: async client wrapper, token accounting,
cost ledger, prompt variants. Two CLIs built on it.

## Status
🚧 Week 1 of 12 · streaming client with cost and latency instrumentation · CLI next

## Setup
​```bash
uv sync
cp .env.example .env   # add your API key
​```

## Usage
​```bash
uv run chat    # streaming multi-turn chatbot
uv run lab     # prompt playground (not yet implemented)
​```

## Design notes

- Every LLM call returns validated usage and computed cost. No exceptions.
- Provider-agnostic: `ProviderConfig` is the only seam. Swapping Gemini → Groq
  is a `.env` change, not a code change.
- Retries are owned here, not by the SDK (`max_retries=0`). Silent SDK retries
  hide rate limiting and inflate latency measurements.
- Unpriced ≠ free. Calls with no pricing entry are counted separately rather
  than recorded as $0.00 — a cost total that can't distinguish the two is lying.
- Free tier bills $0.00, so the ledger tracks *shadow cost*: what each call
  would cost at list price. Same economics, no spend.
- Prompts live in `prompts/*.yaml`, never in Python strings.
- Streaming yields a union of text deltas and a terminal `RunResult`, rather than
  stashing state on the client. Keeps `LLMClient` stateless so concurrent streams
  are safe (needed in week 2's playground).
- `complete()` retries; `stream()` does not. A partially-consumed stream can't be
  replayed without duplicating output for the consumer.

## Baseline measurements

Measured 2026-08-11 · `gemini-3.5-flash` · list pricing $1.50 / $9.00 per Mtok

| Prompt | In | Out | Shadow cost | TTFT | Total | Deltas |
|---|---|---|---|---|---|---|
| Trivial | 2 | 9 | $0.0001 | — | 2446ms | — |
| Short question | 6 | 78 | $0.0007 | — | 2565ms | — |
| Short (streamed) | 5 | 146 | $0.0013 | 3523ms | 3847ms | — |
| Long (streamed) | 5 | 449 | $0.0040 | 8896ms | 10652ms | 18 |

Output tokens are ~99% of cost at this price ratio, and latency scales with
output length. Response length is the lever; prompt length mostly isn't.

## Finding: "streaming" is not one behavior

Gemini's OpenAI-compatible endpoint returned 449 output tokens as 18 batched
deltas — roughly 25 tokens per chunk — with time-to-first-token at 8896ms of a
10652ms total. Technically streaming; functionally, the user waits 83% of the
response time before seeing anything.

Total latency alone would not have surfaced this. Instrumenting delta count and
inter-delta gaps did.