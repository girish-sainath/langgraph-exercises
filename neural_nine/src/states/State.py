"""
This module defines the ChatState class, which represents the state of a chat session.
"""
# pylint: disable=invalid-name
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class State(TypedDict):
    """
    Represents the state of a chat session, including the list of messages exchanged.
    """
    messages: Annotated[list[AnyMessage], add_messages]
    message_intent: str | None
    next_node: str | None
