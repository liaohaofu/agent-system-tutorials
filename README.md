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
from common.llm import call_llm

response = call_llm([{"role": "user", "content": "Hello!"}])
print(response)
```

## Tutorials

| # | Topic | Implementation | Tutorial |
|---|-------|----------------|----------|
| 1 | Structured Output | [implementations/01_structured_output/](implementations/01_structured_output/) | _Coming soon_ |
| 2 | Tool Use | _Coming soon_ | _Coming soon_ |
| 3 | Agent Loop | _Coming soon_ | _Coming soon_ |

_More tutorials coming soon._

## Project Structure

```
agent_system/
├── README.md           # You are here
├── common/             # Shared utilities (LLM wrapper, etc.)
├── implementations/    # Working Python code for each topic
└── tutorials/          # Markdown tutorials explaining concepts
```

## Philosophy

- **No frameworks** — No LangChain, LangGraph, or AutoGen
- **LLM as a component** — The architecture is what we're teaching
- **Everything inspectable** — See every decision the agent makes
- **Minimal first** — Start simple, add complexity only when needed
