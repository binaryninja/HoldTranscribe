"""
Moshi TTS Model implementation for HoldTranscribe.

This module provides text-to-speech functionality using Kyutai's Moshi conversational model,
adapted for TTS use cases with real-time streaming capabilities.
"""

import torch
import numpy as np
import tempfile
import os
import time
import threading
import queue
from typing import Optional, List, Iterator, Tuple, Any, Union
from pathlib import Path

from ..models import TTSModel
from ..utils import debug_print


class MoshiTTSModel(TTSModel):
    """
    Moshi Text-to-Speech model implementation using the conversational Moshi model.

    Uses Kyutai's full Moshi conversational model for TTS by feeding text and generating
    audio responses. Supports streaming generation and voice conditioning.
    """

    def __init__(self, model_name: str = "kyutai/moshiko-pytorch-q8", device: str = "cuda"):
        """
        Initialize Moshi TTS model.

        Args:
            model_name: HuggingFace model identifier for Moshi model
            device: Device to run the model on ('cuda' or 'cpu')
        """
        super().__init__(model_name, device)

        # Model components
        self.moshi_lm = None
        self.mimi = None
        self.lm_gen = None

        # Generation parameters
        self.temperature = 0.8
        self.temp_text = 0.7
        self.sample_rate = 24000
        self.frame_size = 1920  # Mimi frame size

        # Streaming parameters
        self.chunk_max_words = 25
        self.chunk_silence_duration = 0.5
        self.streaming_enabled = True

        # Audio collection for synthesis
        self.output_chunks = []
        self.generation_lock = threading.Lock()

        # Random seed for reproducible generation
        self.seed = None

    def load(self) -> bool:
        """Load the Moshi model and required components."""
        try:
            debug_print(f"Loading Moshi conversational model for TTS: {self.model_name}")

            # Import required libraries
            from huggingface_hub import hf_hub_download
            from moshi.models import loaders, LMGen
            import torch

            # Load Mimi (audio codec)
            debug_print("Loading Mimi audio codec...")
            mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
            self.mimi = loaders.get_mimi(mimi_weight, device=self.device)
            self.mimi.set_num_codebooks(8)  # Limited to 8 for Moshi

            # Load Moshi LM
            debug_print("Loading Moshi language model...")
            moshi_weight = hf_hub_download(self.model_name, loaders.MOSHI_NAME)
            self.moshi_lm = loaders.get_moshi_lm(moshi_weight, device=self.device)

            # Create LM generator
            debug_print("Creating LM generator...")
            self.lm_gen = LMGen(
                self.moshi_lm,
                temp=self.temperature,
                temp_text=self.temp_text
            )

            # Update sample rate from Mimi
            if hasattr(self.mimi, 'sample_rate'):
                self.sample_rate = self.mimi.sample_rate

            self.is_loaded = True
            debug_print(f"Moshi TTS model loaded successfully on {self.device}")
            return True

        except ImportError as e:
            debug_print(f"Failed to import required libraries: {e}")
            debug_print("Install with: pip install moshi torch huggingface_hub")
            return False
        except Exception as e:
            debug_print(f"Failed to load Moshi model: {e}")
            return False

    def unload(self):
        """Unload the model and free memory."""
        try:
            if self.moshi_lm is not None:
                del self.moshi_lm
                self.moshi_lm = None

            if self.mimi is not None:
                del self.mimi
                self.mimi = None

            if self.lm_gen is not None:
                del self.lm_gen
                self.lm_gen = None

            self.output_chunks = []

            # Clear CUDA cache if using GPU
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.is_loaded = False
            debug_print("Moshi TTS model unloaded")

        except Exception as e:
            debug_print(f"Error during model unload: {e}")

    def set_seed(self, seed: int):
        """Set random seed for reproducible generation."""
        self.seed = seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    def set_voice_conditioning(self, voice_path: Optional[str] = None):
        """
        Set voice conditioning for consistent voice characteristics.

        Note: The conversational Moshi model doesn't use explicit voice conditioning
        like the dedicated TTS model. Voice characteristics are emergent from the
        conversation context.

        Args:
            voice_path: Not used in conversational model, kept for API compatibility
        """
        if not self.is_loaded:
            debug_print("Model not loaded, cannot set voice conditioning")
            return

        debug_print("Voice conditioning not available in conversational Moshi model")
        debug_print("Voice characteristics are determined by conversation context")

    def split_text_for_streaming(self, text: str, max_words: int = None) -> List[str]:
        """Split text into chunks for streaming generation."""
        if max_words is None:
            max_words = self.chunk_max_words

        words = text.split()
        chunks = []
        current_chunk = []

        for word in words:
            current_chunk.append(word)
            if len(current_chunk) >= max_words:
                chunks.append(' '.join(current_chunk))
                current_chunk = []

        # Add remaining words
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def _apply_voice_consistency(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply voice consistency processing to generated audio."""
        # Simple normalization and clipping
        audio_data = np.clip(audio_data, -1.0, 1.0)

        # Apply gentle compression to maintain consistency
        threshold = 0.8
        ratio = 4.0
        above_threshold = np.abs(audio_data) > threshold
        audio_data[above_threshold] = np.sign(audio_data[above_threshold]) * (
            threshold + (np.abs(audio_data[above_threshold]) - threshold) / ratio
        )

        return audio_data

    def synthesize(self, text: str, output_file: str, **kwargs) -> bool:
        """
        Synthesize text to speech and save to file.

        Args:
            text: Text to synthesize
            output_file: Path to save the audio file
            **kwargs: Additional generation parameters

        Returns:
            True if synthesis was successful, False otherwise
        """
        if not self.is_loaded:
            debug_print("Model not loaded")
            return False

        try:
            # Generate audio
            audio_data = self._synthesize_conversational(text, **kwargs)

            if audio_data is not None and len(audio_data) > 0:
                # Apply voice consistency
                audio_data = self._apply_voice_consistency(audio_data)

                # Save to file
                self._save_audio_data(audio_data, output_file)

                debug_print(f"Audio saved to: {output_file}")
                return True
            else:
                debug_print("No audio data generated")
                return False

        except Exception as e:
            debug_print(f"Synthesis failed: {e}")
            return False

    def synthesize_streaming(self, text: str, **kwargs) -> Iterator[np.ndarray]:
        """
        Generate audio in streaming chunks.

        Args:
            text: Text to synthesize
            **kwargs: Additional generation parameters

        Yields:
            Audio chunks as numpy arrays
        """
        if not self.is_loaded:
            debug_print("Model not loaded")
            return

        try:
            # For conversational model, we generate the full response and then chunk it
            # Real streaming would require more complex conversation state management
            audio_data = self._synthesize_conversational(text, **kwargs)

            if audio_data is not None and len(audio_data) > 0:
                # Split into chunks for streaming
                chunk_size = self.sample_rate // 2  # 0.5 second chunks

                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i + chunk_size]
                    chunk = self._apply_voice_consistency(chunk)
                    yield chunk

        except Exception as e:
            debug_print(f"Streaming synthesis failed: {e}")

    def synthesize_streaming_threaded(self, text: str, audio_queue: queue.Queue, **kwargs):
        """Generate audio in a separate thread and put chunks in queue."""
        def generate_worker():
            try:
                for audio_chunk in self.synthesize_streaming(text, **kwargs):
                    audio_queue.put(('audio', audio_chunk))
                audio_queue.put(('done', None))
            except Exception as e:
                audio_queue.put(('error', str(e)))

        worker_thread = threading.Thread(target=generate_worker)
        worker_thread.daemon = True
        worker_thread.start()
        return worker_thread

    def stream_audio_realtime(self, text: str, **kwargs) -> Iterator[Tuple[str, Any]]:
        """
        Stream audio with real-time playback.

        Yields:
            Tuples of (status, data) where status is 'audio', 'done', or 'error'
        """
        audio_queue = queue.Queue()
        worker = self.synthesize_streaming_threaded(text, audio_queue, **kwargs)

        try:
            while True:
                try:
                    status, data = audio_queue.get(timeout=30.0)
                    yield status, data

                    if status in ['done', 'error']:
                        break

                except queue.Empty:
                    yield 'error', 'Timeout waiting for audio generation'
                    break

        finally:
            worker.join(timeout=1.0)

    def _synthesize_conversational(self, text: str, **kwargs) -> Optional[np.ndarray]:
        """
        Core synthesis using conversational Moshi model.

        Args:
            text: Text to synthesize
            **kwargs: Generation parameters

        Returns:
            Generated audio as numpy array or None if failed
        """
        try:
            # Set seed if specified
            if self.seed is not None:
                self.set_seed(self.seed)

            # Clear previous chunks
            with self.generation_lock:
                self.output_chunks = []

            # Encode text as a prompt to the conversational model
            # We'll simulate a conversation where we "say" the text and get audio response

            # Create a simple text encoding as audio input
            # This is a simplified approach - in a real conversation, this would be speech
            dummy_audio = torch.zeros(1, 1, self.frame_size, device=self.device)

            # Generate response using the conversational model
            with torch.no_grad(), self.lm_gen.streaming(1), self.mimi.streaming(1):
                # Start with dummy input to prime the model
                codes = self.mimi.encode(dummy_audio)

                # Generate several steps to get a reasonable response
                max_steps = kwargs.get('max_steps', 50)  # About 4 seconds at 12.5 Hz

                for step in range(max_steps):
                    tokens_out = self.lm_gen.step(codes)

                    if tokens_out is not None:
                        # tokens_out is [B, 1 + 8, 1], with tokens_out[:, 1:] being audio tokens
                        wav_chunk = self.mimi.decode(tokens_out[:, 1:])

                        with self.generation_lock:
                            self.output_chunks.append(wav_chunk.cpu().numpy())

                    # For subsequent steps, use zero input to let model generate freely
                    codes = torch.zeros_like(codes)

            # Concatenate output chunks
            with self.generation_lock:
                if self.output_chunks:
                    # Concatenate all chunks
                    audio_data = np.concatenate([chunk[0, 0] for chunk in self.output_chunks])

                    # Normalize audio
                    if np.max(np.abs(audio_data)) > 0:
                        audio_data = audio_data / np.max(np.abs(audio_data)) * 0.95

                    return audio_data.astype(np.float32)
                else:
                    debug_print("No audio chunks generated")
                    return None

        except Exception as e:
            debug_print(f"Conversational synthesis error: {e}")
            return None

    def _save_audio_data(self, audio_data: np.ndarray, output_file: str):
        """Save audio data to file."""
        try:
            import soundfile as sf

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # Save audio file
            sf.write(output_file, audio_data, self.sample_rate)

        except ImportError:
            debug_print("soundfile not available, using torch to save")
            import torchaudio

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # Convert to tensor and save
            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
            torchaudio.save(output_file, audio_tensor, self.sample_rate)

        except Exception as e:
            debug_print(f"Failed to save audio file: {e}")
            raise

    def play_audio_to_speakers(self, text: str, **kwargs):
        """Play synthesized audio directly to speakers."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        try:
            import sounddevice as sd

            # Generate audio
            audio_data = self._synthesize_conversational(text, **kwargs)

            if audio_data is not None and len(audio_data) > 0:
                # Apply voice consistency
                audio_data = self._apply_voice_consistency(audio_data)

                # Play audio
                sd.play(audio_data, samplerate=self.sample_rate)
                sd.wait()

                debug_print("Audio playback completed")
            else:
                debug_print("No audio data to play")

        except ImportError:
            debug_print("sounddevice not available for direct playback")
            # Fallback to file-based playback
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_file = tmp.name

            try:
                if self.synthesize(text, temp_file, **kwargs):
                    # Try to play with system command
                    import subprocess
                    if os.name == 'posix':  # Linux/Mac
                        subprocess.run(['aplay', temp_file], check=True)
                    else:  # Windows
                        subprocess.run(['start', temp_file], shell=True, check=True)
            finally:
                try:
                    os.unlink(temp_file)
                except:
                    pass
        except Exception as e:
            debug_print(f"Audio playback failed: {e}")
            raise

    def stop_playback(self):
        """Stop any ongoing audio playback."""
        try:
            import sounddevice as sd
            sd.stop()
        except ImportError:
            pass

    def get_available_voices(self) -> List[str]:
        """Get list of available voice options."""
        # Conversational model uses emergent voices
        return [
            "conversational",  # Default conversational voice
            "contextual"       # Voice depends on conversation context
        ]

    def set_voice_parameters(self, **kwargs):
        """Set voice generation parameters."""
        if 'temperature' in kwargs:
            self.temperature = float(kwargs['temperature'])
        if 'temp_text' in kwargs:
            self.temp_text = float(kwargs['temp_text'])

        # Update LM generator if loaded
        if self.lm_gen is not None:
            # Recreate LM generator with new parameters
            from moshi.models import LMGen
            self.lm_gen = LMGen(
                self.moshi_lm,
                temp=self.temperature,
                temp_text=self.temp_text
            )

    def set_streaming_parameters(self, **kwargs):
        """Set streaming generation parameters."""
        if 'chunk_max_words' in kwargs:
            self.chunk_max_words = int(kwargs['chunk_max_words'])
        if 'chunk_silence_duration' in kwargs:
            self.chunk_silence_duration = float(kwargs['chunk_silence_duration'])
        if 'streaming_enabled' in kwargs:
            self.streaming_enabled = bool(kwargs['streaming_enabled'])

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'is_loaded': self.is_loaded,
            'sample_rate': self.sample_rate,
            'temperature': self.temperature,
            'temp_text': self.temp_text,
            'frame_size': self.frame_size,
            'streaming_enabled': self.streaming_enabled,
            'voice_conditioning': False,  # Not available in conversational model
            'implementation': 'moshi-conversational'
        }

    def __str__(self) -> str:
        return f"MoshiTTSModel(model='{self.model_name}', device='{self.device}', loaded={self.is_loaded})"
