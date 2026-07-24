# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses `uv` for dependency management.

```bash
# Install dependencies
uv sync

# Run the application
uv run python main.py

# Run with a specific Python version
uv run --python 3.13 python main.py
```

There are no tests or linter configuration defined yet.

## Architecture

This is a LangGraph-based multi-intent chatbot that routes user messages to different agents based on classified intent.

### Graph flow (`src/graphs/build_graph.py`)

Every message enters at `classifier`, which uses structured LLM output to label the intent as one of three values:

- `chat` → `chat_agent` (general conversation) → END
- `knowledge` → `rag_agent` (RAG over a hardcoded in-memory knowledge base) → END
- `code` → `prepare_coding_request` → `accept_coding` → (human-in-the-loop interrupt)
  - `yes` → `coding_agent` (invokes `claude -p` as a subprocess in `workspace/`) → END
  - `no` → END
  - _revised prompt_ → back to `prepare_coding_request` (loop)

The graph is compiled with `InMemorySaver` for conversation memory across turns, keyed by `thread_id` (a UUID generated per session in `main.py`).

### Model abstraction (`src/models/`)

`ModelFactory` creates chat and embedding models, with two access modes controlled by the `MODEL_ACCESS` env var:

- `litellm` — routes through LiteLLM proxy (default when unset)
- anything else (e.g. `hyperspace-ai`, `direct`) — uses LangChain's `init_chat_model` directly

Model names and temperatures are configured in `ModelInfo` (enum), with values read from `.env` at import time. The three tiers are `default`, `advanced`, and `basic`.

### State (`src/states/State.py`)

`State` is a `TypedDict` with:
- `messages` — append-only via LangGraph's `add_messages` reducer
- `message_intent` — set by classifier, drives the first conditional edge
- `next_node` — set by `accept_coding` / `prepare_coding_request`, drives the second conditional edge

### The `workspace/` directory

`prompt_llm_code` runs `claude -p <prompt> --permission-mode acceptEdits` as a subprocess with `cwd=workspace/`. This is where Claude Code operates when the coding agent is invoked. Any files Claude Code creates or edits will be inside `workspace/`.

## Environment setup

Copy `.env` and populate with credentials. Key variables:

| Variable                              | Purpose                                           |
|---------------------------------------|---------------------------------------------------|
| `MODEL_ACCESS`                        | `litellm` for proxy, anything else for direct API |
| `ANTHROPIC_API_KEY`                   | Required for direct Anthropic access              |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL`  | Required for LiteLLM proxy                        |
