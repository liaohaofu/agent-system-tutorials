"""
Structured Output (Manual Approach)

Get reliable JSON from an LLM — the foundation for agent decisions.
This shows how to do it yourself: schema in prompt → parse → validate → retry.
"""

from openai import OpenAI
from pydantic import BaseModel, ValidationError
from typing import Literal


# --- LLM wrapper ---

client = OpenAI()


def call_llm(messages: list[dict], model: str = "gpt-4o-mini") -> str:
    """Call the LLM and return the response text."""
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


# --- Core implementation ---

SCHEMA_PROMPT_TEMPLATE = """
{user_query}

Respond with JSON matching this schema: {schema}
Return only valid JSON, no other text.
""".strip()


def call_llm_with_schema(
    user_query: str,
    schema: type[BaseModel],
    max_retries: int = 3,
) -> BaseModel:
    """
    Call LLM and parse response into a Pydantic model.

    Retries with error feedback if parsing fails.
    """
    schema_json = schema.model_json_schema()
    prompt = SCHEMA_PROMPT_TEMPLATE.format(user_query=user_query, schema=schema_json)
    messages = [{"role": "user", "content": prompt}]

    last_error = None
    for _ in range(max_retries):
        llm_output = call_llm(messages, model="gpt-5")

        try:
            return schema.model_validate_json(llm_output)
        except ValidationError as e:
            last_error = e
            # Add failed attempt to conversation for retry
            messages.append({"role": "assistant", "content": llm_output})
            messages.append(
                {
                    "role": "user",
                    "content": f"Invalid JSON. Error: {e}. Please fix and try again.",
                }
            )

    raise ValueError(f"Failed after {max_retries} attempts. Last error: {last_error}")


# --- Example usage ---


class MultiChoice(BaseModel):
    answer: Literal["A", "B", "C", "D"]


if __name__ == "__main__":
    question = "How many states are in the United States? A: 48 B: 49 C: 50 D: 51"
    result = call_llm_with_schema(question, schema=MultiChoice)
    print(f"Answer: {result.answer}")
