"""
Agent Class Architecture — From Scattered Code to Clean Abstractions

Extracts reusable abstractions from the parallel tools implementation.
This builds on Chapter 4 (Parallel Tools) with key changes:
1. Tool class encapsulates name, description, schema, and execution
2. ToolRegistry provides centralized tool management
3. LLMOutput formalizes the response structure
4. Agent class orchestrates the loop, tool execution, and message handling

By the end, you'll understand what frameworks like LangChain hide from you.
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Literal

from dataclasses import dataclass
from openai import OpenAI
from pydantic import BaseModel, Field


# --- Tool Class ---


@dataclass
class Tool:
    name: str
    description: str
    parameters: type[BaseModel]
    function: Callable

    @classmethod
    def from_function(cls, function: Callable, parameters: type[BaseModel]) -> "Tool":
        return cls(
            name=function.__name__,
            description=str(function.__doc__),
            parameters=parameters,
            function=function,
        )

    def to_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters.model_json_schema()
        }

    def execute(self, parameters_str: str) -> str:
        try:
            parameters_obj = self.parameters.model_validate_json(parameters_str)
            parameters_dict = parameters_obj.model_dump()
        except Exception as e:
            error_message = f"Invalid tool call parameters: {e}"
            return error_message
        
        try:
            tool_response = self.function(**parameters_dict)
            return str(tool_response)
        except Exception as e:
            error_message = f"Tool call failed: {e}"
            return error_message


# --- Tool Registry ---


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def execute(self, name: str, parameters: str) -> str:
        if name in self._tools:
            return self._tools[name].execute(parameters)
        else:
            error_message = f"Tool '{name}' not registered"
            return error_message


    def to_schemas(self) -> list[dict]:
        return [
            self._tools[tool_name].to_schema()
            for tool_name in sorted(self._tools)
        ]


# --- Output Schema ---


class ToolCall(BaseModel):
    """A request from the LLM to call a tool."""
    id: str = Field(description="an unique id")
    type: Literal["tool_call"] = "tool_call"
    name: str = Field(description="name of the tool call")
    parameters: str = Field(description="parameters of the tool with format matching tool's parameters definition")


class Message(BaseModel):
    """A text message from the LLM (thinking or final answer)."""
    type: Literal["message"] = "message"
    text: str = Field(description="content of the message")


class LLMOutput(BaseModel):
    """
    Structured output from the LLM.

    The LLM returns a list of items, each either a message or tool call.
    This allows:
    - Multiple tool calls in parallel
    - Thinking/reasoning alongside tool calls
    - Clear termination (no tool calls = done)
    """
    content: list[Message | ToolCall] = Field(description="a list of responses - each response is either a message or function call")


# --- Agent Class ---


class Agent:
    def __init__(self, tools: list | None = None, model: str="gpt-5.2", max_steps: int=10):
        self._tool_registry = ToolRegistry()
        tools = tools or []
        for tool_dict in tools:
            tool_obj = Tool.from_function(**tool_dict)
            self._tool_registry.register(tool_obj)

        self.model = model
        self.max_steps = max_steps
        self._client = OpenAI()
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        tools_doc = json.dumps(
            self._tool_registry.to_schemas(), indent=2
        )

        system_prompt = f"""You are provided the following tools to use:
{tools_doc}

Please return your response in the following format:
{LLMOutput.model_json_schema()}

RETURN ONLY THE JSON."""

        return system_prompt

    def _call_llm(self, messages: list[dict]) -> str:
        response = self._client.chat.completions.create(model=self.model, messages=messages)
        return str(response.choices[0].message.content)

    def _execute_tools_parallel(self, tool_calls: list[ToolCall]) -> str:
        tool_call_results = []
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self._tool_registry.execute,
                    tool_call.name,
                    tool_call.parameters
                ): tool_call
                for tool_call in tool_calls
            }
            for future in as_completed(futures):
                function_call = futures[future]
                result = future.result()
                tool_call_results.append(
                    {"id": function_call.id, "result": result}
                )
        return f"Tool call results: {json.dumps(tool_call_results, indent=2)}"

    def run(self, user_query) -> str:
        messages = [
            {
                "role": "developer",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": user_query
            }
        ]

        for step in range(self.max_steps):
            llm_output = self._call_llm(messages)
            messages.append(
                {
                    "role": "assistant",
                    "content": llm_output
                }
            )

            try:
                response = LLMOutput.model_validate_json(llm_output)
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

            returned_tool_calls = [
                response_item for response_item in response.content
                if isinstance(response_item, ToolCall)
            ]
            returned_messages = [
                response_item for response_item in response.content
                if isinstance(response_item, Message)
            ]

            # Print agent response
            for response_item in response.content:
                if isinstance(response_item, Message):
                    print(f"[Step {step + 1}] Agent (message):\n\t{response_item.model_dump_json(indent=2)}")
                else:
                    print(f"[Step {step + 1}] Agent (tool_call):\n\t{response_item.model_dump_json(indent=2)}")

            if returned_tool_calls:
                tool_call_results = self._execute_tools_parallel(returned_tool_calls)
                messages.append({
                    "role": "user",
                    "content": tool_call_results
                })
                print(f"[Step {step + 1}] Tool results:\n{'-' * 10}\n{tool_call_results}\n{'-' * 10}\n")
            else:
                final_response = "\n".join(msg.text for msg in returned_messages)
                return final_response
        return "Max step reached"

# --- Example Tools ---

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
    location: str = Field(description="The city name to get weather for")


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
    sender: str = Field(description="Email address of the sender")
    recipient: str = Field(description="Email address of the recipient")
    message: str = Field(description="Content of the email message")


def get_user_profile() -> dict:
    """get the profile of current user
    """
    return {
        "name": "Jack",
        "email": "jack@example.com"
    }


class GetUserProfileParameters(BaseModel):
    pass

# --- Example usage ---

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
    agent = Agent(tools)

    user_query = "How is the weather in Beijing and LA?"
    response = agent.run(user_query)
    print(f"Q: {user_query}\n\nA:\n{response}")

    print("\n\n")

    user_query = "Get the weather in Beijing and LA and send an summary email to me."
    response = agent.run(user_query)
    print(f"\n\nQ: {user_query}\n\nA:\n{response}")
    print(f"Mail box: \n{json.dumps(MAIL_BOX, indent=2)}")