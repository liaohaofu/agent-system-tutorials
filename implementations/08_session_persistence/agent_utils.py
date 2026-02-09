"""
Shared utilities extracted from previous tutorials:
Tool, ToolRegistry, ToolCall, Message, LLMOutput, Session.
"""
import json
import uuid

from pathlib import Path
from datetime import datetime
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
    
@dataclass(kw_only=True)
class FileSession(Session):
    session_dir: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now())
    _messages: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self.save()

    def save(self):
        session_json = self.asjson()
        session_file = Path(self.session_dir, f"{self.id}.json")
        session_file.parent.mkdir(parents=True, exist_ok=True)
        with open(session_file, 'w') as f:
            f.write(session_json)

    def asjson(self):
        session = {
            'id': self.id,
            'created_at': self.created_at.isoformat(),
            'messages': self._messages
        }
        return json.dumps(session)

    @classmethod
    def load(cls, session_id: str, session_dir: str) -> "FileSession":
        session_file = Path(session_dir, f"{session_id}.json")

        if not session_file.exists():
            raise FileNotFoundError(f"Session file {session_file} does not exist.")

        with open(session_file) as f:
            session = json.load(f)
        created_at = datetime.fromisoformat(session["created_at"])
        messages = session["messages"]

        return cls(session_dir=session_dir, id=session_id, created_at=created_at, _messages=messages)

    @classmethod
    def list_sessions(cls, session_dir: str) -> list:
        session_ids = []
        for session_file in Path(session_dir).glob("*.json"):
            session_ids.append(session_file.name)
        
        return session_ids

    def add_message(self, role, content):
        super().add_message(role, content)
        self.save()

    def set_messages(self, messages: list[dict]):
        super().set_messages(messages)
        self.save()