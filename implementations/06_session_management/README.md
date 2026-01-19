# 06 — Session Management

## What This Teaches

- Why Session should be separate from Agent (reusability, single responsibility)
- Session class for message history management
- Agent owns system prompt (agent identity), Session owns conversation state
- Token-based context management with LLM summarization
- Running the same agent across multiple independent sessions

## Key Concepts

1. **Separation of Concerns**
   - Agent = what the agent is (tools, system prompt, behavior)
   - Session = conversation state (message history)

2. **Context Management**
   - Token-based limit checking via API response usage
   - LLM summarization to compact history when over limit
   - System prompt is never compacted (owned by Agent, not Session)

3. **Design Decision: Where does system_prompt live?**
   - Agent owns it — system prompt defines agent identity
   - Session is pure conversation history
   - Context management handled by Agent (knows model, token limits)

## What This Skips (For Later)

- Persistence (save/load sessions to disk or database)
- Conversation branching/forking
- More sophisticated summarization strategies

## Usage

```bash
python main.py
```
