"""
Chat module for prompting the LLM with the graph and messages state.
"""
from typing import Any
from langchain.chat_models import BaseChatModel

from src.models.ModelInfo import ModelInfo
from src.models.ModelFactory import ModelFactory

from src.states.State import State


def prompt_llm_chat(state: State) -> dict[str, Any]:
    """Prompt the LLM with the graph and messages state, returning the LLM's response."""
    model: BaseChatModel = ModelFactory.create_model(
        ModelInfo.DEFAULT_MODEL_TYPE.value,
    )

    messages = [
                   {
                       'role': 'system',
                       'content': 'You are a talkative chatbot for fun. Be nice',
                   }
               ] + state['messages']

    response = model.invoke(messages)
    return {
        'messages':
            [
                {
                    'role': 'assistant',
                    'content': response.content,
                }
            ],
        }


__all__ = [
    'prompt_llm_chat',
]
