"""
This module provides a function to retrieve model information from environment variables.
"""
# pylint: disable=invalid-name

from enum import Enum
import os
from dotenv import load_dotenv



load_dotenv()

class ModelInfo(Enum):
    """
    Enum for model information, including default, advanced, and basic models,
    as well as the default embedding model and temperature settings.
    """
    LITELLM_DEFAULT_MODEL = os.getenv('LITELLM_DEFAULT_MODEL', 'claude-sonnet-4-6')
    LITELLM_ADVANCED_MODEL = os.getenv('LITELLM_ADVANCED_MODEL', 'claude-opus-4-7')
    LITELLM_BASIC_MODEL = os.getenv('LITELLM_BASIC_MODEL', 'claude-haiku-4-5')
    LITELLM_DEFAULT_EMBEDDING_MODEL = os.getenv('LITELLM_DEFAULT_EMBEDDING_MODEL',
                                                'text-embedding-3-large')
    DIRECT_DEFAULT_MODEL = os.getenv('DIRECT_DEFAULT_MODEL', 'claude-sonnet-4-6')
    DIRECT_ADVANCED_MODEL = os.getenv('DIRECT_ADVANCED_MODEL', 'claude-opus-4-7')
    DIRECT_BASIC_MODEL = os.getenv('DIRECT_BASIC_MODEL', 'claude-haiku-4-5')
    DIRECT_DEFAULT_EMBEDDING_MODEL = os.getenv('DIRECT_DEFAULT_EMBEDDING_MODEL',
                                               'text-embedding-3-large')
    DEFAULT_MODEL_TEMPERATURE = float(os.getenv('DEFAULT_MODEL_TEMPERATURE', '0.5'))
    BASIC_MODEL_TEMPERATURE = float(os.getenv('BASIC_MODEL_TEMPERATURE', '0.7'))

    DEFAULT_MODEL_TYPE = 'default'
    BASIC_MODEL_TYPE = 'basic'
    ADVANCED_MODEL_TYPE = 'advanced'
