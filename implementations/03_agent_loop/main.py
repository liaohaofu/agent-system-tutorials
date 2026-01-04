"""
Agent Loop — Manual Approach

A loop that calls the LLM, executes tools, and feeds results back until done.
This shows how to do it yourself: build message history, parse responses, loop until done.
"""

from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal

# --- LLM wrapper ---

client = OpenAI()


def call_llm(messages: list[dict], model: str = "gpt-4o-mini") -> str:
    """Call the LLM and return the response text."""
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


# --- Core implementation ---


class FunctionCall(BaseModel):
    name: str = Field(description="name of the function")
    parameters: str = Field(description="parameters of the function")


class Response(BaseModel):
    type: Literal["text", "function_call"] = Field(description="type of the response")
    content: str | FunctionCall = Field(
        description="content of the response. str for text response, FunctionCall for function_call response"
    )


def run_agent_loop(user_query, tools, model="gpt-5", max_steps=5):
    print("=" * 50)
    print(f"Query: {user_query}")
    print("=" * 50)
    print()

    tools_doc = []
    tools_dict = {}
    for tool in tools:
        tools_doc.append(
            {
                "name": tool["function"].__name__,
                "description": tool["description"],
                "parameters": tool["parameters"].model_json_schema(),
            }
        )
        tools_dict[tool["function"].__name__] = {
            "function": tool["function"],
            "parameters": tool["parameters"],
        }

    messages = [
        {
            "role": "developer",
            "content": f"You are provided with the following tools: {tools_doc}. Please respond in the following format: {Response.model_json_schema()}. ALWAYS RESPOND DIRECTLY",
        },
        {"role": "user", "content": user_query},
    ]

    for iteration in range(1, max_steps + 1):
        llm_output = call_llm(messages, model=model)
        messages.append({"role": "assistant", "content": llm_output})

        # Parse LLM response
        try:
            response = Response.model_validate_json(llm_output)
        except Exception as e:
            error_message = f"You returned invalid response: {str(e)}"
            print(f"[Iteration {iteration}] {error_message}")
            messages.append({"role": "user", "content": error_message})
            continue

        # Final answer - exit loop
        if response.type == "text":
            print(f"[Iteration {iteration}] {response.content}")
            break

        # Function call - execute and continue
        function_name = response.content.name

        # Check if function exists
        if function_name not in tools_dict:
            error_message = f"{function_name} call error: unknown function"
            print(f"[Iteration {iteration}] {error_message}")
            messages.append({"role": "user", "content": error_message})
            continue

        # Validate parameters
        try:
            parameters = tools_dict[function_name]["parameters"].model_validate_json(
                response.content.parameters
            )
        except Exception as e:
            error_message = f"You returned invalid function call parameters: {str(e)}"
            print(f"[Iteration {iteration}] {error_message}")
            messages.append({"role": "user", "content": error_message})
            continue

        # Execute function
        try:
            result = tools_dict[function_name]["function"](**parameters.model_dump())
            tool_message = f"{function_name} returned {result}"
        except Exception as e:
            tool_message = f"{function_name} call error: {str(e)}"

        print(f"[Iteration {iteration}] {tool_message}")
        messages.append({"role": "user", "content": f"[Tool Result] {tool_message}"})

    else:
        print(f"[Iteration {iteration}] Max step reached")

    print()


# --- Example usage ---


class Parameters(BaseModel):
    item: str = Field(description="name of the item (in lower case)")


def get_price(item: str) -> float:
    """Check the unit price of an item, returns price in $"""
    prices = {"apple": 1.5, "banana": 0.75, "orange": 1.0}
    if item in prices:
        return prices[item]
    else:
        raise ValueError(f"{item} not found. Available items: {list(prices.keys())}")


def get_inventory(item: str) -> int:
    """Get inventory of an item"""
    inventory = {"apple": 3, "banana": 10, "orange": 1}
    if item in inventory:
        return inventory[item]
    else:
        raise ValueError(f"{item} not found. Available items: {list(inventory.keys())}")


if __name__ == "__main__":
    tools = [
        {
            "function": get_price,
            "description": "check the unit price of an item, returns price in $",
            "parameters": Parameters,
        },
        {
            "function": get_inventory,
            "description": "check the inventory (count) of an item",
            "parameters": Parameters,
        },
    ]

    run_agent_loop(
        "I have $5, and I want to buy 5 bananas. Is it possible with the current inventory and price?",
        tools=tools,
    )

    run_agent_loop(
        "I have $5, and I want to buy 5 pineapples. Is it possible with the current inventory and price?",
        tools=tools,
    )

    run_agent_loop("Hi", tools=tools)

    run_agent_loop(
        "I have $5, and I want to buy 5 Apples. Is it possible with the current inventory and price?",
        tools=tools,
        model="gpt-4o-mini",
    )
