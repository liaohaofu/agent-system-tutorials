"""
Parallel Tool Execution — Manual Approach

Extends the agent loop to execute multiple independent tools simultaneously.
This builds on Chapter 3 (Agent Loop) with key changes:
1. Schema supports multiple outputs per response (message or function_call)
2. Each function_call has an ID for result correlation
3. Multiple tool calls execute in parallel using ThreadPoolExecutor
4. Loop terminates when response contains no function calls
"""
import json

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal


# --- LLM wrapper ---
client = OpenAI()


def call_llm(messages: list[dict], model: str = "gpt-4o-mini") -> str:
    """Call the LLM and return the response text."""
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


# --- Output Schema ---
class FunctionCall(BaseModel):
    id: str = Field(description="an unique id")
    type: Literal["function_call"] = "function_call"
    name: str = Field(description="name of the function call")
    parameters: str = Field(description="parameters of the function with format matching function's parameters definition")


class Message(BaseModel):
    type: Literal["message"] = "message"
    text: str = Field(description="content of the message")


class Response(BaseModel):
    content: list[Message | FunctionCall] = Field(description="a list of responses - each response is either a message or function call")


# --- Tools ---
def get_weather(location: str) -> str:
    """get weather at the specified location
    """

    if location.lower() == "los angeles":
        return "sunny, 20 degrees"
    elif location.lower() == "beijing":
        return "cloudy, 5 degrees"
    else:
        raise Exception("Unknown location: supported locations are 'los angeles' and 'beijing'")


class GetWeatherParameters(BaseModel):
    location: str


MAIL_BOX = []


def send_email(sender: str, recipient: str, message: str) -> str:
    """send an email from `sender` to `recipient` with `message`
    """
    MAIL_BOX.append(
        {
            "from": sender,
            "to": recipient,
            "message": message
        }
    )
    return "message sent"


class SendEmailParameters(BaseModel):
    sender: str
    recipient: str
    message: str


def get_user_profile() -> dict:
    """get the profile of current user
    """
    return {
        "name": "Jack",
        "email": "jack@example.com"
    }


class GetUserProfileParameters(BaseModel):
    pass


# --- Tools Execution ---
def execute_tool(tools_dict: dict, function_call: FunctionCall) -> str:
    if function_call.name not in tools_dict:
        return f"Unknown tool: {function_call.name}"

    tool = tools_dict[function_call.name]["function"]
    parameters_format = tools_dict[function_call.name]["parameters"]

    try:
        parameters_obj = parameters_format.model_validate_json(function_call.parameters)
        parameters = parameters_obj.model_dump()
    except Exception as e:
        error_message = f"Invalid function call parameters: {e}"
        return error_message
    
    try:
        tool_response = tool(**parameters)
        return str(tool_response)
    except Exception as e:
        error_message = f"Tool call failed: {e}"
        return error_message


# --- Agentic Loop ---
def run_agentic_loop(user_query, tools, model="gpt-5.2", max_steps=10):
    tools_doc = ""
    tools_dict = {}
    for tool in tools:
        tools_doc += json.dumps({
            "name": tool["function"].__name__,
            "description": tool["function"].__doc__,
            "parameters": tool["parameters"].model_json_schema()
        }, indent=2) + "\n"
        tools_dict[tool["function"].__name__] = {
            "function": tool["function"],
            "parameters": tool["parameters"]
        }


    system_prompt = f"""You are provided the following tools to use
    {tools_doc}
    Please return your response in the following format
    {Response.model_json_schema()}
    RETURN THE JSON FORMAT DIRECTLY
    """


    messages = [
        {
            "role": "developer",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    for step in range(max_steps):
        llm_output = call_llm(messages=messages, model=model)
        messages.append(
            {
                "role": "assistant",
                "content": llm_output
            }
        )

        try:
            response = Response.model_validate_json(llm_output)
        except Exception as e:
            error_message = f"Invalid response format: {e}"
            messages.append(
                {
                    "role": "user",
                    "content": error_message
                }
            )

            print(f"[Step {step + 1}] {error_message}\n")
            continue

        returned_function_calls = [
            response_item for response_item in response.content
            if isinstance(response_item, FunctionCall)
        ]
        returned_messages = [
            response_item for response_item in response.content
            if isinstance(response_item, Message)
        ]

        # Print any messages (thinking/reasoning)
        for msg in returned_messages:
            print(f"[Step {step + 1}] Agent: {msg.text}")

        if returned_function_calls:
            function_call_results = []
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(partial(execute_tool, tools_dict), function_call): function_call
                    for function_call in returned_function_calls
                }
                for future in as_completed(futures):
                    function_call = futures[future]
                    result = future.result()
                    function_call_results.append(
                        {"id": function_call.id, "result": result}
                    )
            
            function_call_message = f"Function call results: {json.dumps(function_call_results, indent=2)}"
            messages.append({
                "role": "user",
                "content": function_call_message
            })

            print(f"[Step {step + 1}] Tool results:\n{'-' * 10}\n{function_call_message}\n{'-' * 10}\n")
        else:
            final_response = "\n".join(msg.text for msg in returned_messages)
            return final_response
    return "Max step reached"


if __name__ == "__main__":
    tools = [
        {
            "function": get_weather,
            "parameters": GetWeatherParameters,
        },
        {
            "function": send_email,
            "parameters": SendEmailParameters,
        },
        {
            "function": get_user_profile,
            "parameters": GetUserProfileParameters
        }
    ]

    user_query = "How is the weather in Beijing and LA?"
    response = run_agentic_loop(user_query, tools=tools)
    print(f"Q: {user_query}\n\nA:\n{response}")

    print("\n\n")

    user_query = "Get the weather in Beijing and LA and send an summary email to me."
    response = run_agentic_loop(user_query, tools=tools)
    print(f"\n\nQ: {user_query}\n\nA:\n{response}")
    print(f"Mail box: \n{json.dumps(MAIL_BOX, indent=2)}")