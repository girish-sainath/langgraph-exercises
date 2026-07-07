"""
A simple intent classifier that classifies user messages into predefined intents.
"""
# pylint: disable=invalid-name
from typing import Literal

from pydantic import BaseModel, Field


class IntentClassifier(BaseModel):
    """
    A simple intent classifier that classifies user messages into predefined intents.
    """
    message_intent: Literal['chat', 'knowledge', 'code'] = (
        Field(..., description='Classify whether the user message '
                               'is related to chat, knowledge, or code.'))
