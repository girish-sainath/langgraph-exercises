"""
This module provides a function to classify the intent of the last user message in a chat state.
It uses a model to determine whether the user wants to chat, retrieve knowledge, or change code.
"""
from src.states.IntentClassifier import IntentClassifier
from src.states.State import State
from src.models.ModelInfo import ModelInfo
from src.models.ModelFactory import ModelFactory

def classify_intent(state: State):
    """
    Classifies the intent of the last user message in the chat state.

    Args:
        state (State): The current state of the chat session.
    """
    model = ModelFactory.create_model(model_type=ModelInfo.DEFAULT_MODEL_TYPE.value)
    structured_model = model.with_structured_output(IntentClassifier)
    response = structured_model.invoke(
        [
            {
                'role': 'system',
                'content': 'Determine / classify whether the user wants to '
                           'chat ("chat"), retrieve knowledge ("knowledge"), '
                           'or change code ("code").',
            },
            {
                'role': 'user',
                'content': state['messages'][-1].content,
            },
        ],
    )

    return {
        'message_intent': response.message_intent,
    }


__all__ = [
    'classify_intent'
]
