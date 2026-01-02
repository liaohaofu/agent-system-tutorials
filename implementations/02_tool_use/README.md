# 02 — Tool Use (Single Shot)

## What This Teaches

- How to define a tool with metadata (name, description, JSON schema)
- How the LLM decides which tool to call based on user input
- How to extract and validate structured arguments
- How to execute a tool and return results

## Two Approaches

1. **Manual approach** (`main.py`)
   - Put tool schema in the prompt
   - LLM returns JSON with tool name + arguments
   - Parse, validate, execute

2. **OpenAI native approach** (`main_openai.py`)
   - Use OpenAI's function calling API
   - Cleaner integration, built-in schema validation

## What This Skips (For Later)

- Multiple sequential tool calls → Agent Loop (03)
- Feeding results back to LLM for follow-up reasoning → Agent Loop (03)
- Tool registry with multiple tools → Part 2
- Error handling / retries → Part 5

## Key Concept

```
User → LLM → Tool → User
        ↓
   (picks tool + args)
```

The LLM acts as a "router" — it decides which tool to call and extracts arguments, but does NOT reason about the result. That's the Agent Loop.

## Usage

```bash
# Manual approach
python main.py

# OpenAI native approach
python main_openai.py
```

## Example

```
User: What is the current date and time in LA?
LLM decided to call: get_datetime({"city_name": "America/Los_Angeles"})
Tool result: 2026-01-01 11:30:45

User: What is 2 + 2?
LLM (no tool): 4
```