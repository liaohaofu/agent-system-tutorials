"""
Tool Use — OpenAI Native Approach

Same result as main.py, but using OpenAI's built-in function calling.
The API handles tool selection and argument extraction.
"""

from datetime import datetime
from openai import OpenAI
from pydantic import BaseModel, Field
from zoneinfo import ZoneInfo


client = OpenAI()


def get_datetime(city_name: str = None) -> str:
    DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

    if not city_name:
        return datetime.now().strftime(DATETIME_FORMAT)
    else:
        return datetime.now(ZoneInfo(city_name)).strftime(DATETIME_FORMAT)


class Parameters(BaseModel):
    city_name: str = Field(description="city name in IANA timezone format")


def run_example(user_query: str, tools: list[dict]):
    """Run a single tool-use example and print results."""
    print(f"User: {user_query}")
    response = client.responses.create(model="gpt-5", input=user_query, tools=tools)

    for output in response.output:
        if output.type == "message":
            # LLM answered directly without using a tool
            print(f"LLM (no tool): {output.content[0].text}")
        elif output.type == "function_call":
            print(f"LLM decided to call: {output.name}({output.arguments})")

            if output.name == "get_datetime":
                parameters = Parameters.model_validate_json(output.arguments)
                datetime_str = get_datetime(parameters.city_name)
                print(f"Tool result: {datetime_str}")
            else:
                raise ValueError("LLM returned unknown function")

    print()


if __name__ == "__main__":
    tools = [
        {
            "type": "function",
            "name": "get_datetime",
            "description": "get the current date and time at the specified city",
            "parameters": Parameters.model_json_schema(),
        }
    ]

    # Example 1: LLM should use the tool
    run_example("What is the current date and time in LA?", tools)

    # Example 2: LLM might answer directly (no tool needed)
    run_example("What is 2 + 2?", tools)
