"""
A simple example of a chat application using LangGraph and LangChain.
"""
from uuid import uuid4
from dotenv import load_dotenv

from langgraph.types import Command

from src.graphs.build_graph import build_graph


load_dotenv()


def main():
    """
    Main function to run the chat application. It builds the graph, prompts the user for input,
    and invokes the graph to get responses from the LLM until the user types "exit".
    :return:
    """
    graph = build_graph()

    graph.get_graph().draw_mermaid_png(output_file_path='graph.png')

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

        while '__interrupt__' in graph_response:
            prompt = graph_response['__interrupt__'][0].value
            decision = input(f'\n{prompt}\n')
            graph_response = graph.invoke(Command[str](resume=decision), config=config)
        print('Response from LLM:\n', graph_response['messages'][-1].content)


if __name__ == '__main__':
    main()
