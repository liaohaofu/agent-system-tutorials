# 01 — Structured Output

Get reliable JSON from an LLM. This is the foundation — agents need parseable decisions, not free-form text.

## What This Teaches

- Why free-text LLM responses break agents
- How to get JSON output reliably using Pydantic schemas
- Parsing and validation
- Retry with error feedback when parsing fails

## Files

- `main.py` — Manual approach (schema in prompt → parse → validate → retry)
- `main_openai.py` — OpenAI's native structured output API

## Usage

```bash
# Install the package first (from project root)
pip install -e .

# Run the manual approach
python implementations/01_structured_output/main.py

# Run the OpenAI native approach
python implementations/01_structured_output/main_openai.py
```

## What This Skips (For Later)

- Tool definitions and execution
- The agent loop

## Tutorial

See the full tutorial: [`tutorials/01-structured-output/`](../../tutorials/01-structured-output/)