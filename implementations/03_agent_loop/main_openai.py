"""
Agent Loop — OpenAI Native Approach

Same result as main.py, but using OpenAI's conversation API with tool results.
The API handles message formatting; you handle the loop logic.
"""

from openai import OpenAI
from pydantic import BaseModel, Field


client = OpenAI()


# --- Example usage ---


if __name__ == "__main__":
    pass