"""
This module contains the function to build a simple state graph for a chatbot application.
"""
from typing import Any

from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver

from src.states.State import State
from src.nodes.classify_intent import classify_intent
from src.nodes.prompt_llm_chat import prompt_llm_chat
from src.nodes.prompt_llm_rag import prompt_llm_rag
from src.nodes.prompt_llm_code import prompt_llm_code
from src.nodes.accept_coding import accept_coding
from src.nodes.prepare_coding_request import prepare_coding_request


def build_graph():
    """Build a simple graph with a single LLM prompt node."""
    graph_builder: StateGraph[Any] = StateGraph(State)  # type: ignore[arg-type]
    graph_builder.add_node(
        'classifier',
        classify_intent,  # type: ignore[arg-type]
    )
    graph_builder.add_node(
        'chat_agent',
        prompt_llm_chat,  # type: ignore[arg-type]
    )
    graph_builder.add_node(
        'rag_agent',
        prompt_llm_rag,  # type: ignore[arg-type]
    )
    graph_builder.add_node(
        'accept_coding',
        accept_coding,  # type: ignore[arg-type]
    )
    graph_builder.add_node(
        'prepare_coding_request',
        prepare_coding_request, # type: ignore[arg-type]
    )
    graph_builder.add_node(
        'coding_agent',
        prompt_llm_code,  # type: ignore[arg-type]
    )

    graph_builder.add_edge(
        START,
        'classifier',
    )
    graph_builder.add_edge(
        'prepare_coding_request',
        'accept_coding',
    )
    graph_builder.add_conditional_edges(
        'accept_coding',
        lambda state: 'end'
        if state.get('next_node') == 'denied'
        else state.get('next_node'),
        {
            'end': END,
            'coding_agent': 'coding_agent',
            'revise': 'prepare_coding_request',
        },
    )
    graph_builder.add_conditional_edges(
        'classifier',
        lambda state: state['message_intent'],
        {
            'chat': 'chat_agent',
            'knowledge': 'rag_agent',
            'code': 'prepare_coding_request',
        },
    )
    graph_builder.add_edge(
        'chat_agent',
        END,
    )
    graph_builder.add_edge(
        'rag_agent',
        END,
    )
    graph_builder.add_edge(
        'coding_agent',
        END,
    )

    return graph_builder.compile(
        checkpointer=InMemorySaver(),
    )


__all__ = [
    'build_graph',
]
