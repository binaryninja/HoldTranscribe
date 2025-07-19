"""
Dia TTS model implementation.
"""

import os
import tempfile
import re
import threading
import queue
import random
import numpy as np
from typing import Optional, Any, Generator, List, Tuple

# Try different import methods for Dia
HAS_DIA = False
HAS_DIA_TRANSFORMERS = False

try:
    from transformers import AutoProcessor, DiaForConditionalGeneration
    HAS_DIA = True
    HAS_DIA_TRANSFORMERS = True
except ImportError:
    try:
        from dia.model import Dia
        HAS_DIA = True
        HAS_DIA_TRANSFORMERS = False
    except ImportError:
        HAS_DIA = False
        HAS_DIA_TRANSFORMERS = False

from . import TTSModel
from ..utils import debug_print


class DiaTTSModel(TTSModel):
    """Dia model implementation for text-to-speech synthesis with streaming support."""

    def __init__(self, model_name: str, device: str, use_transformers: bool = None):
        super().__init__(model_name, device)
        self.model = None
        self.processor = None

        # Determine which implementation to use
        if use_transformers is None:
            self.use_transformers = HAS_DIA_TRANSFORMERS
        else:
            self.use_transformers = use_transformers and HAS_DIA_TRANSFORMERS

        if not HAS_DIA:
            raise ImportError("Dia dependencies not available. Install with appropriate Dia package")

        # Set default model name if not specified
        if not model_name or model_name == "default":
            self.model_name = "nari-labs/Dia-1.6B-0626" if self.use_transformers else "default"

        # Speaker consistency settings
        self.fixed_seed = None
        self.audio_prompt_path = None
        self.voice_sample = None
        self.chunk_overlap_speaker = True

        # Streaming settings
        self.chunk_max_words = 50
        self.chunk_silence_duration = 0.5  # seconds of silence between chunks

    def load(self) -> bool:
        """Load the Dia TTS model."""
        try:
            debug_print(f"Loading Dia TTS model: {self.model_name}")

            if self.use_transformers:
                return self._load_transformers_model()
            else:
                return self._load_native_model()

        except Exception as e:
            print(f"⚠️ Failed to load Dia TTS model '{self.model_name}': {e}")
            self.model = None
            self.processor = None
            self.is_loaded = False
            return False

    def _load_transformers_model(self) -> bool:
        """Load Dia model using Transformers library."""
        try:
            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = DiaForConditionalGeneration.from_pretrained(self.model_name)

            # Move to device
            self.model = self.model.to(self.device)

            self.is_loaded = True
            print(f"🔊 Dia TTS model '{self.model_name}' loaded successfully on {self.device}")
            return True

        except Exception as e:
            debug_print(f"Failed to load Transformers Dia model: {e}")
            return False

    def _load_native_model(self) -> bool:
        """Load Dia model using native Dia library."""
        try:
            # Load using native Dia implementation
            self.model = Dia(device=self.device)

            self.is_loaded = True
            print(f"🔊 Dia TTS model loaded successfully on {self.device}")
            return True

        except Exception as e:
            debug_print(f"Failed to load native Dia model: {e}")
            return False

    def unload(self):
        """Unload the Dia TTS model."""
        if self.model is not None:
            if self.use_transformers and hasattr(self.model, 'cpu'):
                if self.device == "cuda":
                    self.model.cpu()
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except ImportError:
                        pass

            self.model = None
            self.processor = None
            self.is_loaded = False
            debug_print(f"Dia TTS model '{self.model_name}' unloaded")

    def set_seed(self, seed: int):
        """Set random seed for consistent voice generation."""
        self.fixed_seed = seed
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    def set_voice_conditioning(self, audio_prompt_path: str = None, voice_sample: np.ndarray = None):
        """Set audio conditioning for consistent voice across chunks."""
        self.audio_prompt_path = audio_prompt_path
        self.voice_sample = voice_sample

    def split_text_for_streaming(self, text: str, max_words: int = None) -> List[str]:
        """
        Split text into chunks optimized for streaming TTS.

        Args:
            text: Input text to split
            max_words: Maximum words per chunk (default: self.chunk_max_words)

        Returns:
            List of text chunks with speaker continuity preserved
        """
        if max_words is None:
            max_words = self.chunk_max_words

        # Split on existing line breaks for dialogue lines
        lines = text.splitlines()
        chunks = []
        current_chunk = []
        word_count = 0
        last_speaker = None

        for line in lines:
            if not line.strip():  # blank line, preserve as paragraph break
                current_chunk.append(line)
                continue

            # Detect speaker tag at line start, e.g. "[S1]" or "[Narrator]"
            speaker_match = re.match(r'^\s*\[([^\]]+)\]', line)

            # If chunk is long and new speaker starts, create new chunk
            if word_count >= max_words and speaker_match:
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []

                # Carry over last speaker tag to maintain context if enabled
                if last_speaker and self.chunk_overlap_speaker:
                    current_chunk.append(f"[{last_speaker}]")
                word_count = 0

            if speaker_match:
                last_speaker = speaker_match.group(1)

            current_chunk.append(line)

            # Update word count
            word_count += len(re.findall(r'\S+', line))

            # If chunk is very large, break at sentence boundary
            if word_count >= max_words * 1.5 and re.search(r'[.!?]\s*$', line):
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    if last_speaker and self.chunk_overlap_speaker:
                        current_chunk.append(f"[{last_speaker}]")
                    word_count = 0

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return [chunk for chunk in chunks if chunk.strip()]

    def _apply_voice_consistency(self, chunk_index: int = 0):
        """Apply voice consistency measures before generation."""
        # Apply fixed seed if set
        if self.fixed_seed is not None:
            self.set_seed(self.fixed_seed)

        # Note: Audio prompt conditioning will be handled in synthesis methods

    def synthesize(self, text: str, output_file: str, streaming: bool = False, **kwargs) -> bool:
        """
        Synthesize text to speech.

        Args:
            text: Text to synthesize
            output_file: Path to save the generated audio
            streaming: Whether to use streaming synthesis for long text
            **kwargs: Additional parameters for synthesis

        Returns:
            True if synthesis was successful
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Dia TTS model is not loaded")

        try:
            if streaming:
                return self._synthesize_streaming(text, output_file, **kwargs)
            else:
                if self.use_transformers:
                    return self._synthesize_transformers(text, output_file, **kwargs)
                else:
                    return self._synthesize_native(text, output_file, **kwargs)

        except Exception as e:
            error_msg = f"TTS synthesis failed: {e}"
            debug_print(error_msg)
            return False

    def synthesize_streaming(self, text: str, **kwargs) -> Generator[np.ndarray, None, None]:
        """
        Generate streaming audio chunks for real-time playback.

        Args:
            text: Text to synthesize
            **kwargs: Additional parameters for synthesis

        Yields:
            Audio data as numpy arrays for each chunk
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Dia TTS model is not loaded")

        # Split text into chunks
        chunks = self.split_text_for_streaming(text)
        debug_print(f"Split text into {len(chunks)} chunks for streaming")

        for i, chunk in enumerate(chunks):
            try:
                # Apply voice consistency
                self._apply_voice_consistency(i)

                # Generate audio for this chunk
                if self.use_transformers:
                    audio_data = self._generate_chunk_transformers(chunk, **kwargs)
                else:
                    audio_data = self._generate_chunk_native(chunk, **kwargs)

                if audio_data is not None:
                    yield audio_data

            except Exception as e:
                debug_print(f"Failed to generate chunk {i}: {e}")
                continue

    def synthesize_streaming_threaded(self, text: str, audio_queue: queue.Queue, **kwargs):
        """
        Generate streaming audio chunks in a background thread for real-time playback.

        Args:
            text: Text to synthesize
            audio_queue: Queue to put generated audio chunks
            **kwargs: Additional parameters for synthesis
        """
        def generate_worker():
            try:
                for audio_chunk in self.synthesize_streaming(text, **kwargs):
                    audio_queue.put(('audio', audio_chunk))
                audio_queue.put(('done', None))
            except Exception as e:
                audio_queue.put(('error', str(e)))

        worker_thread = threading.Thread(target=generate_worker, daemon=True)
        worker_thread.start()
        return worker_thread

    def stream_audio_realtime(self, text: str, **kwargs) -> Generator[Tuple[str, Any], None, None]:
        """
        Stream audio with minimal latency using background generation.

        Args:
            text: Text to synthesize
            **kwargs: Additional parameters for synthesis

        Yields:
            Tuples of (status, data) where status is 'audio', 'done', or 'error'
        """
        audio_queue = queue.Queue(maxsize=2)  # Small buffer for real-time streaming

        # Start background generation
        worker_thread = self.synthesize_streaming_threaded(text, audio_queue, **kwargs)

        try:
            while True:
                try:
                    # Get next chunk with timeout to avoid hanging
                    status, data = audio_queue.get(timeout=30.0)
                    yield status, data

                    if status in ['done', 'error']:
                        break

                except queue.Empty:
                    yield 'error', 'Timeout waiting for audio generation'
                    break

        finally:
            # Clean up
            if worker_thread.is_alive():
                worker_thread.join(timeout=1.0)

    def _synthesize_streaming(self, text: str, output_file: str, **kwargs) -> bool:
        """Synthesize using streaming approach and save to file."""
        try:
            all_audio = []
            chunk_count = 0
            temp_files = []

            for audio_chunk in self.synthesize_streaming(text, **kwargs):
                all_audio.append(audio_chunk)
                chunk_count += 1

            if all_audio:
                if self.use_transformers and self.processor:
                    # Save individual chunks first
                    import tempfile
                    import os

                    for i, audio_chunk in enumerate(all_audio):
                        temp_file = f"temp_chunk_{i}.wav"
                        self.processor.save_audio([audio_chunk], temp_file)
                        temp_files.append(temp_file)

                    # Concatenate audio files using audio processing
                    self._concatenate_audio_files(temp_files, output_file)

                    # Clean up temp files
                    for temp_file in temp_files:
                        try:
                            os.remove(temp_file)
                        except:
                            pass
                else:
                    # For native implementation, concatenate and save
                    final_audio = np.concatenate(all_audio)
                    sample_rate = kwargs.get('sample_rate', 22050)
                    self._save_audio_data(final_audio, output_file, sample_rate)

                debug_print(f"Streaming TTS audio saved to: {output_file} ({chunk_count} chunks)")
                return True
            else:
                debug_print("No audio chunks generated")
                return False

        except Exception as e:
            debug_print(f"Streaming synthesis failed: {e}")
            return False

    def synthesize_streaming_threaded(self, text: str, audio_queue: queue.Queue, **kwargs):
        """
        Generate streaming audio chunks in a background thread for real-time playback.

        Args:
            text: Text to synthesize
            audio_queue: Queue to put generated audio chunks
            **kwargs: Additional parameters for synthesis
        """
        def generate_worker():
            try:
                for audio_chunk in self.synthesize_streaming(text, **kwargs):
                    audio_queue.put(('audio', audio_chunk))
                audio_queue.put(('done', None))
            except Exception as e:
                audio_queue.put(('error', str(e)))

        worker_thread = threading.Thread(target=generate_worker, daemon=True)
        worker_thread.start()
        return worker_thread

    def stream_audio_realtime(self, text: str, **kwargs) -> Generator[Tuple[str, Any], None, None]:
        """
        Stream audio with minimal latency using background generation.

        Args:
            text: Text to synthesize
            **kwargs: Additional parameters for synthesis

        Yields:
            Tuples of (status, data) where status is 'audio', 'done', or 'error'
        """
        audio_queue = queue.Queue(maxsize=2)  # Small buffer for real-time streaming

        # Start background generation
        worker_thread = self.synthesize_streaming_threaded(text, audio_queue, **kwargs)

        try:
            while True:
                try:
                    # Get next chunk with timeout to avoid hanging
                    status, data = audio_queue.get(timeout=30.0)
                    yield status, data

                    if status in ['done', 'error']:
                        break

                except queue.Empty:
                    yield 'error', 'Timeout waiting for audio generation'
                    break

        finally:
            # Clean up
            if worker_thread.is_alive():
                worker_thread.join(timeout=1.0)


    def _generate_chunk_transformers(self, chunk: str, **kwargs) -> Optional[np.ndarray]:
        """Generate audio for a single chunk using Transformers implementation."""
        try:
            import torch

            # Format text for Dia (add speaker tags if not present)
            if not chunk.startswith("[S"):
                formatted_text = f"[S1] {chunk}"
            else:
                formatted_text = chunk

            debug_print(f"Generating chunk: '{formatted_text[:50]}...'")

            # Process text input (note: text must be a list)
            inputs = self.processor(text=[formatted_text], padding=True, return_tensors="pt").to(self.device)

            # Add audio conditioning if available
            generation_kwargs = {
                'max_new_tokens': kwargs.get('max_new_tokens', 3072),
                'guidance_scale': kwargs.get('guidance_scale', 3.0),
                'temperature': kwargs.get('temperature', 1.8),
                'top_p': kwargs.get('top_p', 0.90),
                'top_k': kwargs.get('top_k', 45)
            }

            if self.audio_prompt_path:
                generation_kwargs['audio_prompt_path'] = self.audio_prompt_path

            # Generate audio
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_kwargs)

            # Decode audio
            outputs = self.processor.batch_decode(outputs)

            # Return the decoded audio directly - no need to convert to numpy
            if isinstance(outputs, list) and len(outputs) > 0:
                return outputs[0]

            return None

        except Exception as e:
            debug_print(f"Transformers chunk generation failed: {e}")
            return None

    def _generate_chunk_native(self, chunk: str, **kwargs) -> Optional[np.ndarray]:
        """Generate audio for a single chunk using native Dia implementation."""
        try:
            # Generate audio using native Dia
            generation_kwargs = {
                'text': chunk,
                'voice': kwargs.get('voice', 'default'),
                'speed': kwargs.get('speed', 1.0),
                'pitch': kwargs.get('pitch', 1.0)
            }

            if self.audio_prompt_path:
                generation_kwargs['audio_prompt_path'] = self.audio_prompt_path

            audio_data = self.model.synthesize(**generation_kwargs)

            # Ensure audio_data is numpy array
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data)

            return audio_data

        except Exception as e:
            debug_print(f"Native chunk generation failed: {e}")
            return None

    def _synthesize_transformers(self, text: str, output_file: str, **kwargs) -> bool:
        """Synthesize using Transformers implementation."""
        try:
            # Apply voice consistency
            self._apply_voice_consistency(0)

            audio_data = self._generate_chunk_transformers(text, **kwargs)

            if audio_data is not None:
                # Use processor.save_audio for proper output
                if self.use_transformers and self.processor:
                    self.processor.save_audio([audio_data], output_file)
                else:
                    # Fallback for native implementation
                    sample_rate = kwargs.get('sample_rate', 22050)
                    self._save_audio_data(audio_data, output_file, sample_rate)
                debug_print(f"TTS audio saved to: {output_file}")
                return True
            else:
                debug_print("Failed to generate audio data")
                return False

        except Exception as e:
            debug_print(f"Transformers TTS synthesis failed: {e}")
            return False

    def _synthesize_native(self, text: str, output_file: str, **kwargs) -> bool:
        """Synthesize using native Dia implementation."""
        try:
            # Apply voice consistency
            self._apply_voice_consistency(0)

            audio_data = self._generate_chunk_native(text, **kwargs)

            if audio_data is not None:
                sample_rate = kwargs.get('sample_rate', 22050)
                self._save_audio_data(audio_data, output_file, sample_rate)
                debug_print(f"TTS audio saved to: {output_file}")
                return True
            else:
                debug_print("Failed to generate audio data")
                return False

        except Exception as e:
            debug_print(f"Native TTS synthesis failed: {e}")
            return False

    def _concatenate_audio_files(self, input_files: list, output_file: str):
        """Concatenate multiple audio files into one."""
        try:
            import wave
            import os

            if not input_files:
                return

            # Read first file to get parameters
            with wave.open(input_files[0], 'rb') as first_wav:
                params = first_wav.getparams()
                sample_rate = first_wav.getframerate()
                channels = first_wav.getnchannels()
                sample_width = first_wav.getsampwidth()

            # Open output file
            with wave.open(output_file, 'wb') as output_wav:
                output_wav.setparams(params)

                # Concatenate all input files
                for input_file in input_files:
                    if os.path.exists(input_file):
                        with wave.open(input_file, 'rb') as input_wav:
                            output_wav.writeframes(input_wav.readframes(input_wav.getnframes()))

            debug_print(f"Concatenated {len(input_files)} audio files into {output_file}")

        except Exception as e:
            debug_print(f"Audio concatenation failed: {e}")
            # Fallback: just copy the first file
            if input_files and os.path.exists(input_files[0]):
                import shutil
                shutil.copy2(input_files[0], output_file)

    def _save_audio_data(self, audio_data, output_file: str, sample_rate: int):
        """Save audio data to file."""
        import numpy as np
        import wave

        # Ensure audio_data is numpy array
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data)

        # Convert to 16-bit PCM
        if audio_data.dtype != np.int16:
            # Normalize to [-1, 1] if not already
            if audio_data.max() > 1.0 or audio_data.min() < -1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            # Convert to 16-bit PCM
            audio_data = (audio_data * 32767).astype(np.int16)

        # Save as WAV file
        with wave.open(output_file, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

    def get_available_voices(self) -> list:
        """Get list of available voices."""
        if self.use_transformers:
            # For transformers-based models, this depends on the specific model
            return ["default"]
        else:
            # For native Dia, try to get available voices
            try:
                if hasattr(self.model, 'get_voices'):
                    return self.model.get_voices()
                else:
                    return ["default"]
            except:
                return ["default"]

    def set_voice_parameters(self, voice: str = None, speed: float = 1.0, pitch: float = 1.0):
        """Set voice parameters for synthesis."""
        self.voice = voice or "default"
        self.speed = speed
        self.pitch = pitch

    def set_streaming_parameters(self, chunk_max_words: int = 50, chunk_silence_duration: float = 0.5,
                               chunk_overlap_speaker: bool = True):
        """
        Set parameters for streaming synthesis.

        Args:
            chunk_max_words: Maximum words per chunk for streaming
            chunk_silence_duration: Silence duration between chunks in seconds
            chunk_overlap_speaker: Whether to overlap speaker tags between chunks
        """
        self.chunk_max_words = chunk_max_words
        self.chunk_silence_duration = chunk_silence_duration
        self.chunk_overlap_speaker = chunk_overlap_speaker

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "implementation": "transformers" if self.use_transformers else "native",
            "is_loaded": self.is_loaded,
            "available_voices": self.get_available_voices(),
            "streaming_support": True,
            "voice_consistency": {
                "fixed_seed": self.fixed_seed,
                "audio_prompt": self.audio_prompt_path is not None or self.voice_sample is not None,
                "chunk_overlap_speaker": self.chunk_overlap_speaker
            },
            "streaming_config": {
                "chunk_max_words": self.chunk_max_words,
                "chunk_silence_duration": self.chunk_silence_duration
            }
        }

    def __str__(self) -> str:
        impl = "transformers" if self.use_transformers else "native"
        return f"DiaTTSModel(model={self.model_name}, device={self.device}, impl={impl}, loaded={self.is_loaded})"
