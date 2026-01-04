# 03 — Agent Loop

## What This Teaches

- The core agent pattern: think → act → observe → repeat
- How to feed tool results back to the LLM
- Termination conditions (final answer vs. continue looping)
- Safety limits to prevent infinite loops

## Two Approaches

1. **Manual approach** (`main.py`)
   - Build the message history yourself
   - Parse LLM output to detect tool calls vs. final answers
   - Append tool results to messages and loop

2. **OpenAI Agent SDK approach** (`main_openai.py`)
   - Uses OpenAI's Agent SDK to handle the loop internally
   - Shows how SDKs abstract the loop pattern

## Key Concept

The difference between Tutorial 02 (single-shot tool use) and this tutorial:

| Tutorial 02 | Tutorial 03 |
|-------------|-------------|
| One LLM call | Multiple LLM calls in a loop |
| One tool execution | Multiple tool executions |
| User asks → Tool runs → Done | User asks → Tool runs → LLM sees result → Maybe calls another tool → ... → Done |

## What This Skips (For Later)

- Parallel tool execution
- Error handling / retries
- Planning before acting

## Usage

```bash
# Manual approach
python main.py

# OpenAI Agent SDK approach
python main_openai.py
```

## Expected Output

```
==================================================
Query: I have $5, and I want to buy 5 bananas. Is it possible with the current inventory and price?
==================================================

[Iteration 1] get_price returned 0.75
[Iteration 2] get_inventory returned 10
[Iteration 3] Yes, you can buy 5 bananas! The total cost would be $3.75 (5 × $0.75), which is within your $5 budget, and there are 10 bananas in stock.

==================================================
Query: I have $5, and I want to buy 5 pineapples. Is it possible with the current inventory and price?
==================================================

[Iteration 1] get_price call error: pineapples not found. Available items: ['apple', 'banana', 'orange']
[Iteration 2] Sorry, pineapples are not available. The available items are: apple, banana, and orange.
```