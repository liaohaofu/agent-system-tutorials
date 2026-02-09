# 08 — Session Persistence & Resumption

## What This Teaches

- Persisting session history to disk (JSON files)
- Loading and resuming previous sessions across process restarts
- Handling session metadata (timestamps, IDs)
- Extending `Session` with dataclass inheritance (`kw_only=True`)

## Key Concepts

1. **FileSession extends Session** — Inherits the base interface, adds file persistence
2. **Auto-save on mutation** — Every `add_message` and `set_messages` call persists immediately
3. **Session discovery** — `list_sessions()` to find available sessions, `load()` by ID
4. **Compaction-aware** — Compacted sessions store the summary, loading is transparent

## Architecture

```
Session (base)           FileSession (persistent)
├── add_message()   →    ├── add_message() + save()
├── set_messages()  →    ├── set_messages() + save()
├── get_messages()       ├── get_messages()
└── id                   ├── save() / load() / list_sessions()
                         ├── session_dir
                         └── created_at
```

## What This Skips (For Later)

- Database backends (SQLite, PostgreSQL)
- Cloud storage
- Session search/querying
- Concurrent access handling

## Usage

```bash
python main.py
```
