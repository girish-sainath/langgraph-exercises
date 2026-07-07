"""
Code module for prompting the LLM with the graph and messages state.
"""
import os
import subprocess
from typing import Any

from src.states.State import State


def prompt_llm_code(state: State) -> dict[str, Any]:
    """Prompt the LLM with the graph and messages state, returning the LLM's response."""

    user_prompt = state['messages'][-1].content
    workspace = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'workspace')

    result = subprocess.run(
        ['claude', '-p', user_prompt, '--permission-mode', 'acceptEdits'],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )

    response = result.stdout.strip() or result.stderr.strip()

    return {'messages': [{'role': 'assistant', 'content': response}]}


__all__ = [
    'prompt_llm_code',
]
