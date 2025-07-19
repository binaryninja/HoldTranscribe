"""
ElevenLabs TTS Model wrapper for HoldTranscribe compatibility.
"""

import logging
from typing import Optional, Any
from . import TTSModel, ModelType
from .elevenlabs_model import ElevenLabsTTSModel

logger = logging.getLogger(__name__)


class ElevenLabsTTSWrapper(TTSModel):
    """Wrapper for ElevenLabs TTS model to implement TTSModel interface."""

    def __init__(self, model_name: str, device: str, **kwargs):
        """
        Initialize ElevenLabs TTS wrapper.

        Args:
            model_name: Name/ID of the ElevenLabs model to use
            device: Device parameter (ignored for cloud-based ElevenLabs)
            **kwargs: Additional parameters including api_key
        """
        super().__init__(model_name, device)

        # Extract ElevenLabs-specific parameters
        self.api_key = kwargs.get('api_key')
        self.voice_id = kwargs.get('voice_id', "21m00Tcm4TlvDq8ikWAM")
        self.voice_settings = kwargs.get('voice_settings')
        self.output_format = kwargs.get('output_format', "mp3_44100_128")

        # Initialize the actual ElevenLabs model
        # Remove api_key from kwargs to avoid conflict
        elevenlabs_kwargs = {k: v for k, v in kwargs.items() if k != 'api_key'}
        self.elevenlabs_model = ElevenLabsTTSModel(
            api_key=self.api_key,
            model_id=model_name,
            **elevenlabs_kwargs
        )

        logger.info(f"Initialized ElevenLabs TTS wrapper with model: {model_name}")

    def load(self) -> bool:
        """Load the ElevenLabs model."""
        try:
            success = self.elevenlabs_model.load()
            if success:
                self.is_loaded = True
                logger.info(f"Successfully loaded ElevenLabs model: {self.model_name}")
            else:
                logger.error(f"Failed to load ElevenLabs model: {self.model_name}")
            return success
        except Exception as e:
            logger.error(f"Error loading ElevenLabs model: {e}")
            return False

    def unload(self):
        """Unload the ElevenLabs model."""
        try:
            self.elevenlabs_model.unload()
            self.is_loaded = False
            logger.info(f"Unloaded ElevenLabs model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error unloading ElevenLabs model: {e}")

    def synthesize(self, text: str, output_file: str, **kwargs) -> bool:
        """
        Synthesize text to speech and save to output file.

        Args:
            text: Text to synthesize
            output_file: Path where to save the generated audio
            **kwargs: Additional synthesis parameters

        Returns:
            True if synthesis was successful, False otherwise
        """
        if not self.is_loaded:
            logger.error("ElevenLabs model not loaded")
            return False

        try:
            # Merge instance settings with kwargs
            synthesis_kwargs = {
                'voice_id': kwargs.get('voice_id', self.voice_id),
                'voice_settings': kwargs.get('voice_settings', self.voice_settings),
                'output_format': kwargs.get('output_format', self.output_format),
                **kwargs
            }

            # Call the ElevenLabs model's synthesize method
            result_path = self.elevenlabs_model.synthesize(
                text=text,
                output_path=output_file,
                **synthesis_kwargs
            )

            if result_path:
                logger.info(f"Successfully synthesized text to {output_file}")
                return True
            else:
                logger.error("ElevenLabs synthesis returned no result")
                return False

        except Exception as e:
            logger.error(f"Error during ElevenLabs synthesis: {e}")
            return False

    def synthesize_streaming(self, text: str, **kwargs):
        """
        Synthesize text to speech with streaming.

        Args:
            text: Text to synthesize
            **kwargs: Additional synthesis parameters

        Returns:
            Generator yielding audio chunks or None if failed
        """
        if not self.is_loaded:
            logger.error("ElevenLabs model not loaded")
            return None

        try:
            # Merge instance settings with kwargs
            synthesis_kwargs = {
                'voice_id': kwargs.get('voice_id', self.voice_id),
                'voice_settings': kwargs.get('voice_settings', self.voice_settings),
                'output_format': kwargs.get('output_format', self.output_format),
                **kwargs
            }

            return self.elevenlabs_model.synthesize_streaming(text, **synthesis_kwargs)

        except Exception as e:
            logger.error(f"Error during ElevenLabs streaming synthesis: {e}")
            return None

    def get_available_voices(self):
        """Get list of available voices from ElevenLabs."""
        if not self.is_loaded:
            logger.error("ElevenLabs model not loaded")
            return []

        return self.elevenlabs_model.get_available_voices()

    def set_voice_parameters(self, voice_id: str, **voice_settings):
        """Set voice parameters."""
        self.voice_id = voice_id
        if voice_settings:
            self.voice_settings = voice_settings

        if self.is_loaded:
            self.elevenlabs_model.set_voice_parameters(voice_id, **voice_settings)

    def set_streaming_parameters(self, **streaming_params):
        """Set streaming parameters."""
        if self.is_loaded:
            self.elevenlabs_model.set_streaming_parameters(**streaming_params)

    def get_model_info(self):
        """Get model information."""
        if not self.is_loaded:
            return {
                'name': 'ElevenLabs TTS',
                'loaded': False,
                'model_name': self.model_name,
                'voice_id': self.voice_id
            }

        return self.elevenlabs_model.get_model_info()

    @property
    def model_type(self) -> ModelType:
        """Return the model type."""
        return ModelType.TTS

    def __str__(self) -> str:
        return f"ElevenLabsTTSWrapper(model_name={self.model_name}, loaded={self.is_loaded})"
