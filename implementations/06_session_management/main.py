"""
Session Management — Separating Conversation State from Agent Logic

Extracts conversation history into a dedicated Session class.
This builds on Agent Class Architecture with key changes:
1. Session class owns message history (not Agent)
2. Agent owns system prompt (defines agent identity)
3. Token-based context management with LLM summarization
4. Same agent can run across multiple independent sessions

By the end, you'll understand why frameworks separate Agent from Session.
"""

import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Literal

from dataclasses import dataclass, field
from openai import OpenAI
from openai.types import CompletionUsage
from pydantic import BaseModel, Field


# --- Tool Class (from Agent Class Architecture) ---

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


# --- Tool Registry (from Agent Class Architecture) ---


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


# --- Output Schema (from Agent Class Architecture) ---


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


# --- Session Class ---


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


# --- Agent Class (modified to use Session) ---


class Agent:
    def __init__(
            self, tools: list | None = None, model: str="gpt-5.2", max_steps: int=10,
            max_prompt_tokens: int=100_000
        ):
        self._tool_registry = ToolRegistry()
        tools = tools or []
        for tool_dict in tools:
            tool_obj = Tool.from_function(**tool_dict)
            self._tool_registry.register(tool_obj)

        self.model = model
        self.max_steps = max_steps
        self.max_prompt_tokens = max_prompt_tokens
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

    def _build_messages(self, session: Session):
        return [dict(role="developer", content=self.system_prompt)] + session.get_messages()
    
    def _compact_session(self, session: Session):
        session_messages = session.get_messages()
        compact_messages, _ = self._call_llm(
            [{
                "role": "user",
                "content": f"Session history: {session_messages}\n\nSummarize the session history into a compact one. Return the compact history only."
            }]
        )
        session.set_messages(
            [dict(role="user", content=f"[COMPACT SESSION HISTORY]: {compact_messages}")]
        )

    def _is_over_limit(self, usage: CompletionUsage) -> bool:
        return usage.prompt_tokens >= self.max_prompt_tokens

    def _call_llm(self, messages: list[dict]) -> tuple[str, CompletionUsage]:
        response = self._client.chat.completions.create(model=self.model, messages=messages)
        return str(response.choices[0].message.content), response.usage

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

    def run(self, user_query, session: Session | None = None) -> str:
        if not session:
            session = Session()
        session.add_message(role="user", content=user_query)

        for step in range(self.max_steps):
            llm_output, usage = self._call_llm(self._build_messages(session))
            # Compact AFTER detecting over-limit, not before. This works because:
            # 1. max_prompt_tokens is a soft limit we set, below the model's hard limit
            # 2. The call succeeded, but we're approaching the limit — compact for NEXT iteration
            # 3. The response (llm_output) is still valid and gets added below
            if self._is_over_limit(usage):
                print(f"[Step {step + 1}] Context is over the limit, compacting for next iteration")
                self._compact_session(session)
            session.add_message(role="assistant", content=llm_output)

            try:
                response = LLMOutput.model_validate_json(llm_output)
            except Exception as e:
                error_message = f"Invalid response format: {e}"
                session.add_message(role="user", content=error_message)

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
                session.add_message(role="user", content=tool_call_results)

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
    agent = Agent(tools, max_prompt_tokens=1200)

    user_query = "How is the weather in Beijing and LA?"
    session = Session()
    response = agent.run(user_query, session=session)
    print(f"Q: {user_query}\n\nA:\n{response}")

    print("========Session History========")
    print(json.dumps(session.get_messages(), indent=2))
    print("===============================")


    print("\n\n")

    user_query = "Send an summary of the weather to me via email (use my own email to send the information)"
    response = agent.run(user_query, session=session)
    print(f"\n\nQ: {user_query}\n\nA:\n{response}")
    print(f"Mail box: \n{json.dumps(MAIL_BOX, indent=2)}")

    print("========Session History========")
    print(json.dumps(session.get_messages(), indent=2))
    print("===============================")
