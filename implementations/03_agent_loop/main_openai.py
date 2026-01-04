"""
Agent Loop — OpenAI Native Approach

Same result as main.py, but using OpenAI's Agent SDK with tool results.
"""

from agents import Agent, Runner, function_tool


def run_agent_loop(user_query, tools, model="gpt-5", max_steps=5):
    print("=" * 50)
    print(f"Query: {user_query}")
    print("=" * 50)
    print()

    agent = Agent(
        name="Assistant",
        tools=tools,
        model=model,
    )
    result = Runner.run_sync(agent, user_query, max_turns=max_steps)
    print(result.final_output)
    print()


# --- Example usage ---


@function_tool
def get_price(item: str) -> float:
    """Check the unit price of an item, returns price in $"""
    prices = {"apple": 1.5, "banana": 0.75, "orange": 1.0}
    if item in prices:
        return prices[item]
    else:
        raise ValueError(f"{item} not found. Available items: {list(prices.keys())}")


@function_tool
def get_inventory(item: str) -> int:
    """Get inventory of an item"""
    inventory = {"apple": 3, "banana": 10, "orange": 1}
    if item in inventory:
        return inventory[item]
    else:
        raise ValueError(f"{item} not found. Available items: {list(inventory.keys())}")


if __name__ == "__main__":
    tools = [get_price, get_inventory]

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
