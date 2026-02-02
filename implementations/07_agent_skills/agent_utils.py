"""
Shared utilities extracted from previous tutorials:
Tool, ToolRegistry, ToolCall, Message, LLMOutput, Session.
"""
import uuid

from dataclasses import dataclass, field
from typing import Callable, Literal
from pydantic import BaseModel, Field


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


@dataclass
class Session:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    _messages: list[dict] = field(default_factory=list)

    def add_message(self, role, content):
        self._messages.append(dict(role=role, content=content))

    def set_messages(self, messages: list[dict]):
        self._messages = messages

    def get_messages(self) -> list[dict]:
        return list(self._messages)