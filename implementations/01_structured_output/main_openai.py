"""
Structured Output (OpenAI Native Approach)

Same result as main.py, but using OpenAI's built-in structured output.
This is how it's done in production — the API handles parsing and validation.
"""

from pydantic import BaseModel
from typing import Literal
from openai import OpenAI

client = OpenAI()


def call_llm_with_schema(
    user_query: str,
    schema: type[BaseModel],
    model: str = "gpt-4o-mini",
) -> BaseModel:
    """
    Call LLM with structured output using OpenAI's native support.

    The API guarantees the response matches the schema.
    """
    response = client.responses.parse(
        model=model,
        input=[{"role": "user", "content": user_query}],
        text_format=schema,
    )
    return response.output_parsed


# --- Example usage ---

class MultiChoice(BaseModel):
    answer: Literal["A", "B", "C", "D"]


if __name__ == "__main__":
    question = "How many states are in the United States? A: 48 B: 49 C: 50 D: 51"
    result = call_llm_with_schema(question, schema=MultiChoice)
    print(f"Answer: {result.answer}")
