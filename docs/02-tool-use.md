---
title: "Tool Use"
description: "Let the LLM decide which tool to call — the bridge from chatbot to agent"
order: 2
tags: ["agent", "tutorial", "python", "tools", "function-calling"]
---

# Tool Use

## What You'll Learn

- How to define a tool with metadata (name, description, schema)
- How the LLM decides whether and which tool to call
- How to extract and validate arguments from the LLM's response
- The difference between "LLM as router" vs "LLM as thinker"

## Prerequisites

- [Structured Output](01-structured-output.md) — we need the LLM to return structured JSON
- Python 3.10+
- OpenAI API key configured

## The Concept

In the previous tutorial, we got the LLM to return structured data. Now we use that capability to let the LLM *do things*.

A **tool** is just a function the LLM can call. But the LLM doesn't execute the function directly — it tells us which function to call and with what arguments. We execute it.

```
User → LLM → Tool → User
        ↓
   (picks tool + args)
```

This is "single-shot" tool use. The LLM acts as a **router**: it decides which tool to call and extracts the arguments, but it doesn't reason about the result. That's what the Agent Loop (next tutorial) adds.

**Key insight:** The LLM might not always use a tool. If the user asks "What is 2 + 2?", the LLM can answer directly without calling anything.

## Key Implementation

### 1. Define a tool with metadata

A tool needs three things:

- **Name** — what to call it
- **Description** — when to use it (this is what the LLM reads!)
- **Parameters schema** — what arguments it accepts

```python
class Parameters(BaseModel):
    city_name: str = Field(description="city name in IANA timezone format")

tools = [
    {
        "name": "get_datetime",
        "description": "Get the current date and time at specified city",
        "parameters": Parameters.model_json_schema(),
    }
]
```

The description is crucial — it's how the LLM knows when to use the tool. Bad descriptions lead to wrong tool choices.

### 2. Define the response format

The LLM needs to tell us: did it call a tool, or did it answer directly?

```python
class FunctionCall(BaseModel):
    name: str = Field(description="Name of the tool")
    parameters: str = Field(description="parameters of the tool")

class Response(BaseModel):
    type: Literal["text", "function_call"]
    content: str | FunctionCall
```

When `type` is `"text"`, the LLM answered directly. When it's `"function_call"`, we have a tool to execute.

### 3. Ask the LLM to decide

```python
def call_llm_with_tools(user_query, tools):
    response_format = Response.model_json_schema()
    messages = [
        {
            "role": "developer",
            "content": f"You have the following tools available: {tools}. "
                       f"Respond in this format: {response_format}",
        },
        {"role": "user", "content": user_query},
    ]

    llm_output = call_llm(messages)
    return Response.model_validate_json(llm_output)
```

### 4. Execute the tool (if called)

```python
response = call_llm_with_tools(user_query, tools)

if response.type == "text":
    # LLM answered directly
    print(f"LLM: {response.content}")
elif response.type == "function_call":
    function_call = response.content
    print(f"LLM decided to call: {function_call.name}({function_call.parameters})")

    # Execute the tool
    if function_call.name == "get_datetime":
        params = Parameters.model_validate_json(function_call.parameters)
        result = get_datetime(params.city_name)
        print(f"Tool result: {result}")
```

The key here: we print what the LLM decided before executing. This makes the decision visible and debuggable.

## Full Implementation

See complete code: [`implementations/02_tool_use/`](https://github.com/liaohaofu/agent-system-tutorials/tree/main/implementations/02_tool_use)

- [`main.py`](https://github.com/liaohaofu/agent-system-tutorials/blob/main/implementations/02_tool_use/main.py) — Manual approach (tool schema in prompt, parse response, execute)
- [`main_openai.py`](https://github.com/liaohaofu/agent-system-tutorials/blob/main/implementations/02_tool_use/main_openai.py) — OpenAI's native function calling API

## OpenAI Native Approach

Just like with structured output, OpenAI provides a native API for tool calling:

```python
tools = [
    {
        "type": "function",
        "name": "get_datetime",
        "description": "get the current date and time at the specified city",
        "parameters": Parameters.model_json_schema(),
    }
]

response = client.responses.create(
    model="gpt-5",
    input=user_query,
    tools=tools
)

for output in response.output:
    if output.type == "message":
        print(f"LLM: {output.content[0].text}")
    elif output.type == "function_call":
        print(f"Calling: {output.name}({output.arguments})")
```

The API handles parsing and validation. You just provide the tool definitions.

## What This Doesn't Do (Yet)

This tutorial covers **single-shot** tool use:

- One user query → one tool call (or direct answer) → done

What's missing:

- **Multiple tool calls in sequence** — "Calculate 25 * 17, then add 50"
- **Feeding results back to the LLM** — letting it reason about what the tool returned
- **Deciding when to stop** — knowing the task is complete

That's what the Agent Loop adds. The loop takes tool results and feeds them back to the LLM, which can then decide to call another tool or give a final answer.

## Try It Yourself

- [ ] Run `main.py` and observe when the LLM uses the tool vs answers directly
- [ ] Add a second tool (e.g., `calculator`) and see if the LLM picks correctly
- [ ] Try a query where the LLM might be unsure — does it use the tool or guess?
- [ ] Modify the tool description to be vague — what breaks?

## What's Next

We can now call tools, but only once per query. To handle "Calculate this, then do that", we need the LLM to see tool results and decide what to do next. That's the **Agent Loop**: [Agent Loop](03-agent-loop.md).
