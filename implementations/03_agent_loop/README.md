# 03 — Agent Loop

## What This Teaches

- The core agent pattern: think → act → observe → repeat
- How to feed tool results back to the LLM
- Termination conditions (final answer vs. continue looping)
- Safety limits to prevent infinite loops

## Building On

This tutorial reuses the `get_datetime` tool from Tutorial 02.
The tool code is duplicated here so this example runs standalone.

## Two Approaches

1. **Manual approach** (`main.py`)
   - Build the message history yourself
   - Parse LLM output to detect tool calls vs. final answers
   - Append tool results to messages and loop

2. **OpenAI native approach** (`main_openai.py`)
   - Use OpenAI's conversation API with `tool` role messages
   - API handles message formatting; you handle the loop

## Key Concept

The difference between Tutorial 02 (single-shot tool use) and this tutorial:

| Tutorial 02 | Tutorial 03 |
|-------------|-------------|
| One LLM call | Multiple LLM calls in a loop |
| One tool execution | Multiple tool executions |
| User asks → Tool runs → Done | User asks → Tool runs → LLM sees result → Maybe calls another tool → ... → Done |

## What This Skips (For Later)

- Multiple different tools → Tutorial 04 (Tool Registry)
- Error handling / retries → Part 5 (Reliability)
- Planning before acting → Part 3 (Autonomous Planning)

## Usage

```bash
# Manual approach
python main.py

# OpenAI native approach
python main_openai.py
```

## Expected Output

```
User: What time is it in Tokyo and in LA?

[Loop iteration 1]
LLM calls: get_datetime(Tokyo)
Tool result: 2024-01-15 09:30:00

[Loop iteration 2]
LLM calls: get_datetime(LA)
Tool result: 2024-01-14 16:30:00

[Loop iteration 3]
LLM returns final answer: "It's 9:30 AM in Tokyo and 4:30 PM in LA."

Final answer: It's 9:30 AM in Tokyo and 4:30 PM in LA.
```