"""
Minimal LLM wrapper using OpenAI SDK.

This keeps agent code focused on the loop, not the API.
"""

import os
from openai import OpenAI

# Initialize client (uses OPENAI_API_KEY env var by default)
client = OpenAI()


def call_llm(messages: list[dict], model: str = "gpt-5-mini") -> str:
    """
    Call the LLM and return the response text.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        model: Model to use (default: gpt-5-mini for cost efficiency)

    Returns:
        The assistant's response text
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content
