"""
Model interfaces and factory for HoldTranscribe.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Tuple
from enum import Enum


class ModelType(Enum):
    """Types of models supported by HoldTranscribe."""
    TRANSCRIPTION = "transcription"
    ASSISTANT = "assistant"
    TTS = "tts"


class BaseModel(ABC):
    """Base abstract class for all models."""

    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device
        self.is_loaded = False

    @abstractmethod
    def load(self) -> bool:
        """Load the model. Returns True if successful."""
        pass

    @abstractmethod
    def unload(self):
        """Unload the model to free memory."""
        pass

    @property
    @abstractmethod
    def model_type(self) -> ModelType:
        """Return the type of this model."""
        pass


class TranscriptionModel(BaseModel):
    """Base class for transcription models."""

    @property
    def model_type(self) -> ModelType:
        return ModelType.TRANSCRIPTION

    @abstractmethod
    def transcribe(self, audio_data: Any, **kwargs) -> str:
        """Transcribe audio data to text."""
        pass


class AssistantModel(BaseModel):
    """Base class for AI assistant models."""

    @property
    def model_type(self) -> ModelType:
        return ModelType.ASSISTANT

    @abstractmethod
    def generate_response(self, audio_data: Any, prompt: Optional[str] = None, **kwargs) -> str:
        """Generate a response from audio input."""
        pass


class TTSModel(BaseModel):
    """Base class for text-to-speech models."""

    @property
    def model_type(self) -> ModelType:
        return ModelType.TTS

    @abstractmethod
    def synthesize(self, text: str, output_file: str, **kwargs) -> bool:
        """Synthesize text to speech. Returns True if successful."""
        pass


class ModelFactory:
    """Factory for creating models based on configuration."""

    @staticmethod
    def create_transcription_model(model_name: str, device: str, **kwargs) -> Optional[TranscriptionModel]:
        """Create a transcription model."""
        from .whisper_model import WhisperTranscriptionModel

        if "whisper" in model_name.lower() or model_name in ["base", "large-v3"]:
            return WhisperTranscriptionModel(model_name, device, **kwargs)

        return None

    @staticmethod
    def create_assistant_model(model_name: str, device: str, **kwargs) -> Optional[AssistantModel]:
        """Create an AI assistant model."""
        from .voxtral_model import VoxtralAssistantModel

        if "voxtral" in model_name.lower():
            return VoxtralAssistantModel(model_name, device, **kwargs)

        return None

    @staticmethod
    def create_tts_model(model_name: str, device: str, **kwargs) -> Optional[TTSModel]:
        """Create a text-to-speech model."""
        from .dia_model import DiaTTSModel
        from .moshi_model import MoshiTTSModel
        from .elevenlabs_wrapper import ElevenLabsTTSWrapper
        from ..tts.app_wrapper import AppTTSWrapper

        # ElevenLabs models (now default)
        if ("elevenlabs" in model_name.lower() or "eleven" in model_name.lower() or
              model_name.startswith("eleven_") or model_name == "default"):
            # Convert "default" to actual ElevenLabs model name
            actual_model_name = "eleven_multilingual_v2" if model_name == "default" else model_name
            return ElevenLabsTTSWrapper(actual_model_name, device, **kwargs)
        # Moshi/Kyutai models
        elif ("kyutai" in model_name.lower() or "moshi" in model_name.lower() or
              model_name.startswith("kyutai/")):
            return MoshiTTSModel(model_name, device, **kwargs)
        # DIA models
        elif "dia" in model_name.lower() or model_name.startswith("facebook/"):
            return DiaTTSModel(model_name, device, **kwargs)
        # Legacy app wrapper (fallback)
        elif model_name == "app-wrapper":
            return AppTTSWrapper(model_name, device)

        return None

    @staticmethod
    def create_model(model_type: ModelType, model_name: str, device: str, **kwargs) -> Optional[BaseModel]:
        """Create a model based on type."""
        if model_type == ModelType.TRANSCRIPTION:
            return ModelFactory.create_transcription_model(model_name, device, **kwargs)
        elif model_type == ModelType.ASSISTANT:
            return ModelFactory.create_assistant_model(model_name, device, **kwargs)
        elif model_type == ModelType.TTS:
            return ModelFactory.create_tts_model(model_name, device, **kwargs)

        return None


# Model registry for easy access
class ModelRegistry:
    """Registry to manage loaded models."""

    def __init__(self):
        self._models = {}

    def register(self, key: str, model: BaseModel):
        """Register a model with a key."""
        self._models[key] = model

    def get(self, key: str) -> Optional[BaseModel]:
        """Get a model by key."""
        return self._models.get(key)

    def unregister(self, key: str):
        """Unregister a model."""
        if key in self._models:
            model = self._models[key]
            model.unload()
            del self._models[key]

    def clear(self):
        """Unload and clear all models."""
        for model in self._models.values():
            model.unload()
        self._models.clear()

    def list_models(self) -> dict:
        """List all registered models."""
        return {key: model.model_name for key, model in self._models.items()}


# Global model registry instance
model_registry = ModelRegistry()
