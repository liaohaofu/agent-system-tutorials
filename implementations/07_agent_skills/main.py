"""
Agent Skills — On-Demand Knowledge for Agents

The problem: stuffing every domain into the system prompt wastes tokens
and degrades instruction following. Skills solve this with progressive disclosure:
- Tier 1: metadata always present (~100 tokens per skill in tool description)
- Tier 2: full instructions loaded on demand (via load_skill tool)
- Tier 3: resource files loaded only when referenced

The insight: a skill is a tool that returns instructions instead of data.
No new primitives — just another tool registered in the existing Agent.

Builds on Agent class and ToolRegistry from the Agent Class Architecture tutorial.
"""
import json
import re
import yaml

from dataclasses import dataclass
from openai import OpenAI
from openai.types import CompletionUsage
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from agent_utils import BaseModel, Message, Tool, ToolCall, ToolRegistry, Session, LLMOutput, Field


# --- Skill definition (parsed from SKILL.md with YAML frontmatter) ---
@dataclass
class Skill:
    name: str
    description: str
    content: str


# --- SkillRegistry (directory scanning, precedence, deduplication) ---
class SkillRegistry:
    def __init__(self, skill_dir: str | Path | None=None):
        self._skills: dict[str, Skill] = {}

        if skill_dir:
            self.register(skill_dir)
    
    def register(self, skill_dir: str | Path):
        skill_dir = Path(skill_dir)
        parser = re.compile(r"^---\s*(.*?)\s*---\s*(.*)$", re.DOTALL)

        for skill_file in skill_dir.glob("*/SKILL.md"):
            with open(skill_file) as f:
                skill_str = f.read()

            matched = parser.match(skill_str)
            if matched:
                skill_header_str, skill_content_str = matched.groups()
                try:
                    skill_header = yaml.safe_load(skill_header_str)
                    skill = Skill(
                        name=skill_header['name'],
                        description=skill_header['description'],
                        content=skill_content_str
                    )
                    self._skills[skill.name] = skill
                except Exception as e:
                    error_message = f"Warning: failed to parse {skill_file}: {e}"
                    print(error_message)

    def create_load_skill_tool(self):
        def load_skill(skill_name):
            skill = self._skills.get(skill_name)
            if skill:
                return skill.content
            else:
                error_message = f"Skill {skill_name} not found. Available skills: {list(self._skills.keys())}"
                raise Exception(error_message)
        load_skill.__doc__ = self._get_load_skill_doc()
        
        class LoadSkillParameters(BaseModel):
            skill_name: str = Field(description="Skill name")

        return Tool.from_function(function=load_skill, parameters=LoadSkillParameters)

    @property
    def num_skills(self):
        return len(self._skills)

    def _get_load_skill_doc(self):
        skill_doc = "Load a skill. A skill describes what to do when certain scenarios are met. The available skills are:\n"
        for _, skill in self._skills.items():
            skill_description = f"## Skill: {skill.name}\n{skill.description}"
            skill_doc += f"\n\n{skill_description}"

        return skill_doc


# --- Integration with Agent ---

class Agent:
    def __init__(
            self, tools: list | None = None, model: str="gpt-5.2", max_steps: int=10,
            max_prompt_tokens: int=100_000, skill_dir: str | None = None
        ):
        self._tool_registry = ToolRegistry()
        tools = tools or []
        for tool_dict in tools:
            tool_obj = Tool.from_function(**tool_dict)
            self._tool_registry.register(tool_obj)

        if skill_dir:
            skill_register = SkillRegistry(skill_dir)
            if skill_register.num_skills:
                self._tool_registry.register(skill_register.create_load_skill_tool())

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


def read_file(file_path: str) -> str:
    """read file from a given path into string
    """

    with open(file_path) as f:
        return f.read()


class ReadFileParameters(BaseModel):
    file_path: str = Field(description="Path to the file to read")


# --- Example usage ---

if __name__ == "__main__":
    tools = [
        {
            "function": get_weather,
            "parameters": GetWeatherParameters,
        },
        {
            "function": read_file,
            "parameters": ReadFileParameters,
        },
    ]
    agent = Agent(tools, skill_dir="./skills/")

    user_query = "How is the weather in Beijing and LA?"
    session = Session()
    response = agent.run(user_query, session=session)
    print(f"Q: {user_query}\n\nA:\n{response}")


    print("\n\n")

    user_query = "Create a report of the weather."
    response = agent.run(user_query, session=session)
    print(f"\n\nQ: {user_query}\n\nA:\n{response}")
