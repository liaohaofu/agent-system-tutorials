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

2. **OpenAI native approach** (`openai_native.py`)
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
python openai_native.py
```

## Example

```
User: "What's 25 * 17?"
LLM decides: calculator(expression="25 * 17")
Tool returns: 425
Output to user: 425
```