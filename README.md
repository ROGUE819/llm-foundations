# llm-foundations

Shared core for LLM application work: async client wrapper, token accounting,
cost ledger, prompt variants. Two CLIs built on it.

## Status
🚧 In progress — week 1 of 12.

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
- Prompts live in `prompts/*.yaml`, never in Python strings.