"""
RAG module for prompting the LLM with the graph and messages state.
"""
from typing import Any
from langchain.chat_models import BaseChatModel

from langchain_core.vectorstores import VectorStore, InMemoryVectorStore
from langchain_core.documents import Document

from src.models.ModelInfo import ModelInfo
from src.models.ModelFactory import ModelFactory

from src.states.State import State


KNOWLEDGE = [
    ('NeuralNine is a YouTube channel that provides tutorials and educational '
     'content on programming, machine learning, and artificial intelligence.'),
    ('LangChain is a framework for developing applications powered by language models, '
     'enabling developers to build conversational AI and other language-based applications.'),
    ('LangGraph is a tool that allows users to create and visualize knowledge graphs, '
     'which are structured representations of information and relationships between entities.'),
    ('A StateGraph in LangGraph is a representation of the current state of a system or '
     'application, capturing the relationships and interactions between different '
     'components or entities.'),
    ('Checkpointers like InMemorySaver lets LangGraph save the state of a StateGraph in memory, '
     'allowing for easy retrieval and manipulation of the graph data during runtime.'),
    ('RAG (Retrieval-Augmented Generation) is a technique that combines retrieval of '
     'relevant information from a knowledge base with generation of responses by a language model, '
     'enhancing the model\'s ability to provide accurate and contextually relevant answers.'),
]


def _get_vector_store() -> VectorStore:
    """Create and return a vector store for the knowledge base.

    Returns:
        A Chroma vector store containing the knowledge base documents.
    """
    vector_store = InMemoryVectorStore(ModelFactory.create_embedding_model())
    vector_store.add_documents([Document(page_content=content) for content in KNOWLEDGE])

    return vector_store


def _retrieve_relevant_documents(query: str, vector_store: VectorStore = None) -> list[Document]:
    """Retrieve relevant documents from the knowledge base based on the query.

    Args:
        query: The query string to search for relevant documents.
        vector_store: An optional vector store to use for retrieval.
        If not provided, a new vector store will be created.
    """
    if not vector_store:
        vector_store = _get_vector_store()

    documents = vector_store.similarity_search(query, k=3)
    return documents


def prompt_llm_rag(state: State) -> dict[str, Any]:
    """Prompt the LLM with the graph and messages state, returning the LLM's response."""

    model: BaseChatModel = ModelFactory.create_model(
        ModelInfo.DEFAULT_MODEL_TYPE.value,
    )

    vector_store = _get_vector_store()
    documents = _retrieve_relevant_documents(state['messages'][-1].content, vector_store)

    context = '\n'.join(f'- {doc.page_content}' for doc in documents)

    messages = [
                   {
                       'role': 'system',
                       'content': f'You are a RAG agent. '
                                  f'Answer the user using only the context below. '
                                  f'If the answer is not in it, say you don\'t know.'
                                  f'\n\nContext:\n{context}',
                   },
               ] + state['messages']

    response = model.invoke(messages)
    return {
        'messages': [
            {
                'role': 'assistant',
                'content': response.content
            }
        ]
    }


__all__ = [
    'prompt_llm_rag',
]
