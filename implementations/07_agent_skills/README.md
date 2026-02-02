# 07 — Agent Skills

## What This Teaches

- Why stuffing all knowledge into the system prompt doesn't scale
- Skills as progressive disclosure: metadata → instructions → resources
- A skill is a tool that returns instructions, not a new primitive
- SkillRegistry: directory scanning, precedence, deduplication
- `load_skill` as a single tool with dynamic description

## Key Components

1. **Skill definition** — Parsed from SKILL.md files (YAML frontmatter + markdown body)
2. **SkillRegistry** — Scans directories, registers skills by name (last-write-wins for deduplication)
3. **`load_skill` tool** — Registered like any other tool; description lists available skills
4. **Example skill** — A sample skill with a `references/` file demonstrating tier-3 loading

## What This Skips (For Later)

- Composing multiple skills simultaneously
- Skill conflicts and resolution
- Security / permission scoping
- Cross-platform portability

## Usage

```bash
python main.py
```
