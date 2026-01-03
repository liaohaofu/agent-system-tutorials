# AI Agent System — From First Principles

Learn to build AI agents from scratch. No frameworks, no magic — just Python and an LLM.

> **Status:** Work in progress. Tutorials are being added.

## What You'll Learn

- How agent loops work (and why they matter)
- Tool interfaces and execution
- Planning and task decomposition
- Memory and context management
- Error handling and self-correction

## Setup

### 1. Clone and install

```bash
git clone <repo-url>
cd agent_system
pip install -e .
```

### 2. Set your OpenAI API key

```bash
export OPENAI_API_KEY="your-key-here"
```

### 3. Verify it works

```python
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## Tutorials

| # | Topic |
|---|-------|
| 1 | [Structured Output](docs/01-structured-output.md) |
| 2 | [Tool Use](docs/02-tool-use.md) |
| 3 | Agent Loop — _Coming soon_ |

## Project Structure

```
agent-system-tutorials/
├── README.md           # You are here
├── docs/               # Tutorial markdown files
└── implementations/    # Self-contained Python code for each topic
```

## Philosophy

- **No frameworks** — No LangChain, LangGraph, or AutoGen
- **LLM as a component** — The architecture is what we're teaching
- **Everything inspectable** — See every decision the agent makes
- **Minimal first** — Start simple, add complexity only when needed
