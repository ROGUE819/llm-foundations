# llm-foundations

Shared core for LLM application work: async client wrapper, token accounting,
cost ledger, prompt variants. Two CLIs built on it.

## Status
🚧 Week 1 of 12 · client, pricing, and ledger complete · streaming next

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

## Baseline measurements

Measured 2026-08-11 · `gemini-3.5-flash` · list pricing $1.50 / $9.00 per Mtok

| Prompt | In | Out | Shadow cost | Latency |
|---|---|---|---|---|
| Trivial | 2 | 9 | $0.0001 | 2446ms |
| Short question | 6 | 78 | $0.0007 | 2565ms |
| Longer answer | 5 | 150 | $0.0014 | 3002ms |

Output tokens account for ~99% of cost at this price ratio, and latency scales
roughly linearly with output length — the third call has fewer input tokens than
the second but costs 2× and takes 20% longer. Response length is the lever;
prompt length mostly isn't.