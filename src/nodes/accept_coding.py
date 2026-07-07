"""
This module defines the `accept_coding` function,
which prompts the user to approve or reject a coding request
based on the last message in the state.
"""
from langgraph.types import interrupt

from src.states.State import State


def accept_coding(state: State):
    """
    This function prompts the user to approve or reject a coding request
    based on the last message in the state.
    It returns a dictionary indicating the next node to transition to based on the user's decision.
    :param state:
    :return:
    """
    user_prompt = state['messages'][-1].content
    decision = interrupt(f'About to run Claude Code with the following prompt:'
                         f'\n{user_prompt}\n\nApprove? (yes/no, or type a revised request)')

    text = str(decision).strip().lower()

    if text in ['y', 'yes', 'approve', 'ok']:
        return {'next_node': 'coding_agent'}

    if text in [
        'n',
        'no',
        'deny',
        'reject',
        'cancel'
    ]:
        return {
            'messages':
                [
                    {
                        'role': 'assistant',
                        'content': 'Coding request rejected.'
                    }
                ],
            'next_node': 'denied',
        }

    return {
        'messages':
            [
                {
                    'role': 'user',
                    'content': text,
                }
            ],
        'next_node': 'revise',
    }


__all__ = [
    'accept_coding',
]
