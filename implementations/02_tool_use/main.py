"""
Tool Use — Manual Approach

LLM decides which tool to call based on user input.
This shows how to do it yourself: schema in prompt → parse → execute.
"""


from datetime import datetime
from pydantic import BaseModel, Field
from zoneinfo import ZoneInfo
from typing import Literal
from common import call_llm


class FunctionCall(BaseModel):
    name: str = Field(description="Name of the tool")
    parameters: str = Field(description="parameters of the tool")


class Response(BaseModel):
    type: Literal["text", "function_call"] = Field(
        description="type of the response",
    )
    content: str | FunctionCall = Field(description="content of the response")


def call_llm_with_tools(user_query, tools):
    response_format = Response.model_json_schema()
    messages = [
        {
            "role": "developer",
            "content": f"You have the following tools available to use: {tools}. Please give your answer in the following format: {response_format}. ALWAYS GIVE YOUR RESPONSE DIRECTLY in the required format",
        },
        {"role": "user", "content": user_query},
    ]

    llm_output = call_llm(messages, model="gpt-5")
    response = Response.model_validate_json(llm_output)
    return response


# --- Example usage ---


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
    response = call_llm_with_tools(user_query, tools=tools)

    if response.type == "text":
        # LLM answered directly without using a tool
        print(f"LLM (no tool): {response.content}")
    elif response.type == "function_call":
        function_call = response.content
        print(f"LLM decided to call: {function_call.name}({function_call.parameters})")

        if function_call.name == "get_datetime":
            parameters = Parameters.model_validate_json(function_call.parameters)
            datetime_str = get_datetime(parameters.city_name)
            print(f"Tool result: {datetime_str}")
        else:
            raise ValueError("LLM returned unknown function name")
    else:
        raise ValueError("LLM returned unknown response type")

    print()


if __name__ == "__main__":
    tools = [
        {
            "name": "get_datetime",
            "description": "Get the current date and time at specified city",
            "parameters": Parameters.model_json_schema(),
        }
    ]

    # Example 1: LLM should use the tool
    run_example("What is the current date and time in LA?", tools)

    # Example 2: LLM might answer directly (no tool needed)
    run_example("What is 2 + 2?", tools)
