"""
Voxtral assistant model implementation.
"""

import tempfile
import wave
from typing import Optional, Any

try:
    from transformers import AutoProcessor, VoxtralForConditionalGeneration
    import torch
    HAS_VOXTRAL = True
except ImportError:
    HAS_VOXTRAL = False

from . import AssistantModel
from ..utils import debug_print


class VoxtralAssistantModel(AssistantModel):
    """Voxtral model implementation for AI assistant functionality."""

    def __init__(self, model_name: str, device: str, torch_dtype=None):
        super().__init__(model_name, device)
        self.torch_dtype = torch_dtype or torch.bfloat16
        self.model = None
        self.processor = None

        if not HAS_VOXTRAL:
            raise ImportError("Voxtral dependencies not available. Install with: pip install transformers torch")

        if not model_name.startswith("mistralai/Voxtral"):
            raise ValueError(f"Invalid Voxtral model name: {model_name}")

    def load(self) -> bool:
        """Load the Voxtral model and processor."""
        try:
            debug_print(f"Loading Voxtral model: {self.model_name}")

            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.model_name)

            # Load model
            self.model = VoxtralForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype
            ).to(self.device)

            self.is_loaded = True
            print(f"🚀 Voxtral model '{self.model_name}' loaded successfully on {self.device}")
            return True

        except Exception as e:
            print(f"⚠️ Failed to load Voxtral model '{self.model_name}': {e}")
            self.model = None
            self.processor = None
            self.is_loaded = False
            return False

    def unload(self):
        """Unload the Voxtral model."""
        if self.model is not None:
            # Move model to CPU and clear CUDA cache if using GPU
            if self.device == "cuda":
                self.model.cpu()
                torch.cuda.empty_cache()

            self.model = None
            self.processor = None
            self.is_loaded = False
            debug_print(f"Voxtral model '{self.model_name}' unloaded")

    def generate_response(self, audio_data: Any, prompt: str = None, **kwargs) -> str:
        """
        Generate a response from audio input.

        Args:
            audio_data: Can be a file path (str) or numpy array
            prompt: Optional text prompt to guide the response
            **kwargs: Additional parameters for generation

        Returns:
            Generated response text
        """
        if not self.is_loaded or self.model is None or self.processor is None:
            raise RuntimeError("Voxtral model is not loaded")

        try:
            # Handle different types of audio data
            if isinstance(audio_data, str):
                # File path - read audio file
                audio_file = audio_data
            else:
                # Assume numpy array - need to save to temp file
                audio_file = self._save_audio_to_temp_file(audio_data, kwargs.get('sample_rate', 16000))

            # Use Voxtral conversation format
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio",
                            "path": audio_file,
                        },
                    ],
                }
            ]

            # Process with Voxtral using apply_chat_template
            inputs = self.processor.apply_chat_template(conversation)
            inputs = inputs.to(self.device, dtype=torch.bfloat16)

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get('max_new_tokens', 500),
                    temperature=kwargs.get('temperature', 0.0)
                )
                decoded_outputs = self.processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
                response = decoded_outputs[0].strip()

            debug_print(f"Generated response: {len(response)} characters")
            return response

        except Exception as e:
            error_msg = f"Response generation failed: {e}"
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

    def set_system_prompt(self, system_prompt: str):
        """Set a system prompt for consistent behavior."""
        self.system_prompt = system_prompt

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "torch_dtype": str(self.torch_dtype),
            "is_loaded": self.is_loaded,
            "parameters": sum(p.numel() for p in self.model.parameters()) if self.model else 0
        }

    def __str__(self) -> str:
        return f"VoxtralAssistantModel(model={self.model_name}, device={self.device}, loaded={self.is_loaded})"
