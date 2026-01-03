"""
Agent Loop — Manual Approach

A loop that calls the LLM, executes tools, and feeds results back until done.
This shows how to do it yourself: build message history, parse responses, loop until done.
"""

from openai import OpenAI
from pydantic import BaseModel, Field


# --- LLM wrapper ---

client = OpenAI()


def call_llm(messages: list[dict], model: str = "gpt-4o-mini") -> str:
    """Call the LLM and return the response text."""
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


# --- Core implementation ---


# --- Example usage ---


if __name__ == "__main__":
    pass