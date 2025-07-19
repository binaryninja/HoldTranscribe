"""
Whisper transcription model implementation.
"""

import tempfile
import wave
from typing import Optional, Any

from faster_whisper import WhisperModel

from . import TranscriptionModel
from ..utils import debug_print


class WhisperTranscriptionModel(TranscriptionModel):
    """Whisper model implementation for transcription."""

    def __init__(self, model_name: str, device: str, fast_mode: bool = False, beam_size: int = 5):
        super().__init__(model_name, device)
        self.fast_mode = fast_mode
        self.beam_size = beam_size
        self.model = None

        # Map model names to actual Whisper model names
        self._whisper_model_name = self._get_whisper_model_name(model_name)

    def _get_whisper_model_name(self, model_name: str) -> str:
        """Map input model name to actual Whisper model name."""
        if model_name in ["base", "large-v3", "small", "medium", "large", "large-v2"]:
            return model_name
        elif self.fast_mode:
            return "base"
        else:
            return "large-v3"

    def load(self) -> bool:
        """Load the Whisper model."""
        try:
            debug_print(f"Loading Whisper model: {self._whisper_model_name}")

            # Determine compute type based on device
            compute_type = "float16" if self.device == "cuda" else "int8"

            self.model = WhisperModel(
                self._whisper_model_name,
                device=self.device,
                compute_type=compute_type
            )

            self.is_loaded = True
            print(f"🖥️ Whisper model '{self._whisper_model_name}' loaded on {self.device}")
            return True

        except Exception as e:
            print(f"✗ ERROR loading Whisper model: {e}")
            self.model = None
            self.is_loaded = False
            return False

    def unload(self):
        """Unload the Whisper model."""
        if self.model is not None:
            # faster-whisper doesn't have explicit cleanup, but we can clear the reference
            self.model = None
            self.is_loaded = False
            debug_print(f"Whisper model '{self._whisper_model_name}' unloaded")

    def transcribe(self, audio_data: Any, language: str = "auto", **kwargs) -> str:
        """
        Transcribe audio data to text.

        Args:
            audio_data: Can be a file path (str) or numpy array
            language: Language code or "auto" for auto-detection
            **kwargs: Additional parameters for transcription

        Returns:
            Transcribed text string
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Whisper model is not loaded")

        try:
            # Handle different types of audio data
            if isinstance(audio_data, str):
                # File path
                audio_file = audio_data
            else:
                # Assume numpy array - need to save to temp file
                audio_file = self._save_audio_to_temp_file(audio_data, kwargs.get('sample_rate', 16000))

            # Set language parameter
            language_param = None if language == "auto" else language

            # Remove sample_rate from kwargs as faster-whisper doesn't accept it
            transcribe_kwargs = {k: v for k, v in kwargs.items() if k != 'sample_rate'}

            # Perform transcription
            segments, info = self.model.transcribe(
                audio_file,
                beam_size=self.beam_size,
                language=language_param,
                **transcribe_kwargs
            )

            # Combine all segments into a single text
            transcription = " ".join([segment.text for segment in segments]).strip()

            debug_print(f"Transcription completed: {len(transcription)} characters")
            debug_print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

            return transcription

        except Exception as e:
            error_msg = f"Transcription failed: {e}"
            debug_print(error_msg)
            raise RuntimeError(error_msg)

    def _save_audio_to_temp_file(self, audio_data, sample_rate: int) -> str:
        """Save numpy audio data to a temporary WAV file."""
        import numpy as np

        # Ensure audio_data is numpy array
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data)

        # Convert to 16-bit PCM if needed
        if audio_data.dtype != np.int16:
            # Normalize to [-1, 1] if not already
            if audio_data.max() > 1.0 or audio_data.min() < -1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            # Convert to 16-bit PCM
            audio_data = (audio_data * 32767).astype(np.int16)

        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_file.close()

        # Write WAV file
        with wave.open(temp_file.name, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        return temp_file.name

    def get_supported_languages(self) -> list:
        """Get list of supported languages."""
        # This is a static list for Whisper models
        return [
            "auto", "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl",
            "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da",
            "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te",
            "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne",
            "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af",
            "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk",
            "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba",
            "jw", "su"
        ]

    def __str__(self) -> str:
        return f"WhisperTranscriptionModel(model={self._whisper_model_name}, device={self.device}, loaded={self.is_loaded})"
