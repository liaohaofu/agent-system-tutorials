# AI Agent System — From First Principles

Welcome! This tutorial series teaches you how to build AI agents from scratch — no frameworks, no magic, just Python and an LLM.

## Who This Is For

You should be comfortable with:

- Python (functions, classes, basic error handling)
- APIs (making HTTP requests, working with JSON)
- Basic LLM concepts (prompts, completions, tokens)

No prior agent-building experience required. That's what we're here to learn.

## Philosophy

- **No frameworks** — No LangChain, LangGraph, or AutoGen. We build everything ourselves so you understand what's actually happening.
- **LLM as a component** — The model is just one piece. The architecture around it is what makes an agent.
- **Everything inspectable** — You'll see every decision, tool call, and state change.
- **Minimal first** — Start with the simplest working version, add complexity only when needed.

## What You'll Learn

By the end of this series, you'll understand:

- How agent loops work (and why they matter)
- Tool interfaces and execution
- Planning and task decomposition
- Memory and context management
- Error handling and self-correction

More importantly, you'll be able to build these systems yourself — and debug them when they break.

## Tutorials

| # | Topic | What You'll Build |
|---|-------|-------------------|
| 1 | [Structured Output](01-structured-output.md) | Type-safe LLM responses |
| 2 | [Tool Use](02-tool-use.md) | LLM that calls functions |
| 3 | [Agent Loop](03-agent-loop.md) | Autonomous task execution |
| 4 | [Parallel Tools](04-parallel-tools.md) | Concurrent tool execution |
| 5 | [Agent Class](05-agent-class.md) | Reusable agent architecture |
| 6 | [Session Management](06-session-management.md) | Conversation state and context management |

## How to Use This Series

1. **Read the tutorial** — Understand the concept and why it matters
2. **Study the implementation** — Each tutorial links to complete, runnable code
3. **Run it yourself** — Clone the repo, modify the code, break things
4. **Build something** — Apply what you learned to your own project

If you can't implement it yourself, you haven't truly understood it. That's the bar we're aiming for.

## Get Started

Ready? Start with [Structured Output](01-structured-output.md) — the foundation for everything that follows.