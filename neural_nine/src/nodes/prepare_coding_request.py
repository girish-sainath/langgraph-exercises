"""
This module contains the function `prepare_coding_request`,
which prepares a coding request for Claude Code by
rewriting the latest user message into a clear instruction.
"""
from langchain.chat_models import BaseChatModel

from src.states.State import State
from src.models.ModelInfo import ModelInfo
from src.models.ModelFactory import ModelFactory


def prepare_coding_request(state: State):
    """
    Prepares a coding request for Claude Code by rewriting the latest
    user message into a clear instruction.
    :param state: The current state of the chat session,
    including the list of messages exchanged.
    :return:
    """
    model: BaseChatModel = ModelFactory.create_model(
        ModelInfo.DEFAULT_MODEL_TYPE.value,
    )
    messages = [
        {
            'role': 'system',
            'content': 'Rewrite the latest user coding request into a clear instruction '
                       'for Claude Code. '
                       'Use the conversation history as context. Only output the instruction, '
                       'do not include any additional text or explanation.'
        }
    ] + state['messages']

    response = model.invoke(messages)
    return {
        'messages':
            [
                {
                    'role': 'user',
                    'content': response.content,
                }
            ],
        'next_node': 'prepare',
    }
