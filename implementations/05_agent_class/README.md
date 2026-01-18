# 05 — Agent Class Architecture

## What This Teaches

- Extracting reusable abstractions from scattered code
- Creating a self-describing `Tool` class with `to_schema()` and `execute()`
- Centralizing tool management with `ToolRegistry`
- Orchestrating the agent loop with an `Agent` class
- Understanding what frameworks like LangChain hide from you

## The Pattern

```
Tool                    # Self-describing tool abstraction
├── from_function()     # Factory to create from function + params
├── to_schema()         # Generate JSON schema for LLM
└── execute()           # Validate params and run function

ToolRegistry            # Centralized tool management
├── register()          # Add a tool
├── execute()           # Look up and execute by name
└── to_schemas()        # Generate all schemas for system prompt

Agent                   # Orchestrates everything
├── __init__()          # Register tools, build system prompt
├── run()               # Main entry point
├── _build_system_prompt()
├── _call_llm()
└── _execute_tools_parallel()
```

## What's Included

- `Tool` dataclass with schema generation and execution
- `ToolRegistry` for centralized management
- `LLMOutput` schema (Message, ToolCall)
- `Agent` class with parallel tool execution
- Example tools: get_weather, send_email, get_user_profile

## What's Skipped (For Later)

- Session management (message history) → Chapter 2.3
- Streaming responses → Chapter 2.4
- Persistent memory → Part 5
- Error recovery / replanning → Part 6

## Usage

```bash
python main.py
```
