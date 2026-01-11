# 04 — Parallel Tool Execution

## What This Teaches

- How to extend the agent loop to support multiple tool calls per response
- Schema design: per-output `type` field (`message` or `function_call`) instead of top-level response type
- Tool call IDs for correlating results back to requests
- Parallel execution using `ThreadPoolExecutor`
- Termination: loop ends when response contains no function calls

## Building on Chapter 3

This tutorial extends the Agent Loop from Chapter 3. Key changes:

| Aspect | Chapter 3 (Agent Loop) | Chapter 4 (Parallel Tools) |
|--------|------------------------|---------------------------|
| Schema | Single `response.type` field | Per-output `type` field on each item |
| Tool calls | One at a time | Multiple in parallel |
| Execution | Sequential | `ThreadPoolExecutor` |
| Result format | Single result | Results keyed by ID |

## What This Skips (For Later)

- **Error recovery** — What if 2/3 tools fail? → Future tutorial
- **Dynamic parallelism** — LLM deciding parallel vs sequential → Future tutorial
- **Async patterns** — `asyncio` for I/O-bound tools → Future tutorial
- **Dependency detection** — Automatically detecting independent calls → Future tutorial

## Usage

```bash
python main.py
```

## Example Output

```
Q: How is the weather in Beijing and LA?

[Step 1] Agent: I'll check the weather in both cities for you.
Thinking...
----------
Function call results: [
  {
    "id": "weather_beijing",
    "result": "cloudy, 5 degrees"
  },
  {
    "id": "weather_la",
    "result": "sunny, 20 degrees"
  }
]
----------

[Step 2] Agent: Here's the weather: Beijing is cloudy at 5 degrees, LA is sunny at 20 degrees.

A:
Here's the weather: Beijing is cloudy at 5 degrees, LA is sunny at 20 degrees.
```

The second example demonstrates a multi-step workflow:
1. Get weather for both cities (parallel)
2. Get user profile (to find email address)
3. Send summary email

```
Q: Get the weather in Beijing and LA and send an summary email to me.

[Step 1] Agent: I'll get the weather and your profile first.
...
[Step 2] Agent: Now I'll send you the summary email.
...
[Step 3] Agent: Done! I've sent the weather summary to your email.

Mail box:
[
  {
    "from": "weather-bot@example.com",
    "to": "jack@example.com",
    "message": "Weather summary: Beijing is cloudy at 5 degrees, LA is sunny at 20 degrees."
  }
]
```