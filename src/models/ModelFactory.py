"""
This module defines the ModelFactory class, which is responsible for
creating instances of different models based on the provided model type.
"""
# pylint: disable=invalid-name

import os
from typing import Optional

from dotenv import load_dotenv

from langchain_litellm import ChatLiteLLM
from langchain_litellm import LiteLLMEmbeddings  # pylint: disable=no-name-in-module
from langchain.chat_models import init_chat_model, BaseChatModel

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings


from src.models.ModelInfo import ModelInfo

load_dotenv()


class ModelFactory:  # pylint: disable=too-few-public-methods
    """
    A factory class for creating model instances based on the provided model name.
    """

    @staticmethod
    def create_model(
            model_type: str,
            temperature: Optional[float] = None,
            streaming: bool = False,
    ) -> BaseChatModel:
        """
        Create and return an instance of the specified model.

        Args:
            :param model_type: The type of the model.
            :param temperature: Optional temperature setting for the model.
            :param streaming: Whether to enable streaming for the model.
        """

        model_access = os.getenv('MODEL_ACCESS', 'litellm')
        model: str = ''

        match model_type:
            case 'default':
                model = ModelInfo.LITELLM_DEFAULT_MODEL.value \
                    if model_access == 'litellm' \
                    else ModelInfo.DIRECT_DEFAULT_MODEL.value
                temperature = temperature or ModelInfo.DEFAULT_MODEL_TEMPERATURE.value
            case 'advanced':
                model = ModelInfo.LITELLM_ADVANCED_MODEL.value \
                    if model_access == 'litellm' \
                    else ModelInfo.DIRECT_ADVANCED_MODEL.value
            case 'basic':
                model = ModelInfo.LITELLM_BASIC_MODEL.value \
                    if model_access == 'litellm' \
                    else ModelInfo.DIRECT_BASIC_MODEL.value
                temperature = temperature or ModelInfo.BASIC_MODEL_TEMPERATURE.value
            case 'embedding':
                model = ModelInfo.LITELLM_DEFAULT_EMBEDDING_MODEL.value \
                    if model_access == 'litellm' \
                    else ModelInfo.DIRECT_DEFAULT_EMBEDDING_MODEL.value
                temperature = temperature or ModelInfo.DEFAULT_MODEL_TEMPERATURE.value
            case _:
                raise ValueError(f"Unknown model type: {model_type}")

        if model_access == 'litellm':
            print('Using LiteLLM model:', model)
            return ChatLiteLLM(
                model=model,
                temperature=temperature,
                streaming=streaming,
        )

        print('Using Direct model:', model)
        return init_chat_model(
            model=model,
            temperature=temperature,
            streaming=streaming,
        )


    @staticmethod
    def create_embedding_model() -> Embeddings:
        """
        Create and return an instance of the embedding model.
        """
        model_access = os.getenv('MODEL_ACCESS', 'litellm')
        embedding_model = LiteLLMEmbeddings(model=ModelInfo.LITELLM_DEFAULT_EMBEDDING_MODEL.value) \
            if model_access == 'litellm' \
            else OpenAIEmbeddings(model=ModelInfo.DIRECT_DEFAULT_EMBEDDING_MODEL.value)

        return embedding_model
