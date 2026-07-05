"""
A simple example of a chat application using LangGraph and LangChain.
"""
from dataclasses import dataclass, field
from typing import Annotated, Any
from uuid import uuid4
from dotenv import load_dotenv


from langchain_core.messages import AnyMessage
from langchain_litellm import ChatLiteLLM
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver


load_dotenv()


@dataclass
class ChatState:
    """State class for the chat application, holding the list of messages."""
    messages: Annotated[list[AnyMessage], add_messages] = field(default_factory=list)


def prompt_llm(state: ChatState) -> dict[str, list[AnyMessage]]:
    """Prompt the LLM with the graph and messages state, returning the LLM's response."""
    model = ChatLiteLLM(
        model='sap/anthropic--claude-4.6-sonnet',
        temperature=0.1,
    )
    response = model.invoke(state.messages)
    return {'messages': [response]}


def build_graph():
    """Build a simple graph with a single LLM prompt node."""
    graph_builder: Any = StateGraph(ChatState)  # type: ignore[arg-type]
    graph_builder.add_node('prompt_llm', prompt_llm)
    graph_builder.add_edge(START, 'prompt_llm')
    graph_builder.add_edge('prompt_llm', END)

    checkpointer = InMemorySaver()
    return graph_builder.compile(checkpointer=checkpointer)


def main():
    """
    Main function to run the chat application. It builds the graph, prompts the user for input,
    and invokes the graph to get responses from the LLM until the user types "exit".
    :return:
    """
    graph = build_graph()

    print('Programming Assistant...')
    print('Type "exit" to quit...')
    config = {'configurable': {'thread_id': uuid4()}}
    while True:
        user_message = input('\nEnter your message: ')
        if user_message.lower() == 'exit':
            break
        graph_response = graph.invoke(
            {
                'messages': [
                    {
                        'role': 'user',
                        'content': user_message
                    }
                ]
            },
            config=config,
        )
        print('Response from LLM:\n', graph_response['messages'][-1].content)


if __name__ == '__main__':
    main()
