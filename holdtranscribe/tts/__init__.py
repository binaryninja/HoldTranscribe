"""
Text-to-Speech (TTS) module for HoldTranscribe.

This module provides TTS model implementations for converting text to speech
with streaming support and real-time audio playback.
"""

from typing import Dict, Type, Optional, Any

# Import base TTS model
from ..models import TTSModel

# Import available TTS models
from .kyutai_simple import KyutaiSimpleTTS

# Available TTS model implementations
AVAILABLE_MODELS: Dict[str, Type[TTSModel]] = {
    'kyutai': KyutaiSimpleTTS,
    'moshi': KyutaiSimpleTTS,  # Alias for kyutai
}

__all__ = [
    'TTSModel',
    'KyutaiSimpleTTS',
    'AVAILABLE_MODELS',
    'create_tts_model',
    'list_available_models',
]


def create_tts_model(
    model_type: str,
    model_name: str = "default",
    device: str = "cuda",
    **kwargs
) -> Optional[TTSModel]:
    """
    Create a TTS model instance.

    Args:
        model_type: Type of TTS model ('kyutai', 'moshi')
        model_name: Name/path of the model to load
        device: Device to run the model on ('cuda', 'cpu')
        **kwargs: Additional model-specific parameters

    Returns:
        TTS model instance or None if creation fails
    """
    if model_type not in AVAILABLE_MODELS:
        print(f"❌ Unknown TTS model type: {model_type}")
        print(f"Available models: {list(AVAILABLE_MODELS.keys())}")
        return None

    try:
        model_class = AVAILABLE_MODELS[model_type]
        return model_class(model_name=model_name, device=device, **kwargs)
    except Exception as e:
        print(f"❌ Failed to create {model_type} TTS model: {e}")
        return None


def list_available_models() -> Dict[str, str]:
    """
    List all available TTS model types and their descriptions.

    Returns:
        Dictionary mapping model type to description
    """
    return {
        'kyutai': 'Kyutai/Moshi TTS models with CLI-based reliable synthesis',
        'moshi': 'Alias for Kyutai TTS models',
    }


def get_model_requirements(model_type: str) -> Dict[str, Any]:
    """
    Get the requirements for a specific model type.

    Args:
        model_type: Type of TTS model

    Returns:
        Dictionary with requirements information
    """
    requirements = {
        'kyutai': {
            'dependencies': ['torch', 'moshi>=0.2.10', 'sphn', 'sounddevice', 'soundfile'],
            'python_version': '>=3.12',
            'gpu_recommended': True,
            'description': 'Kyutai/Moshi TTS models with reliable CLI-based synthesis'
        },
        'moshi': {
            'dependencies': ['torch', 'moshi>=0.2.10', 'sphn', 'sounddevice', 'soundfile'],
            'python_version': '>=3.12',
            'gpu_recommended': True,
            'description': 'Kyutai/Moshi TTS models with reliable CLI-based synthesis'
        }
    }

    return requirements.get(model_type, {})
