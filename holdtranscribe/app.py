#!/usr/bin/env python3
"""
HoldTranscribe Application - Refactored modular version.

Hold the chosen hot‑key(s) to record → transcribe → copy to clipboard.
"""

import os
import sys
import time
import argparse
import tempfile
import pyperclip
import notify2
from typing import Optional, Union

from .models import ModelFactory, ModelType, model_registry, TranscriptionModel, AssistantModel, TTSModel
from .audio import AudioRecorder, AudioUtils
from .utils import set_debug, debug_print, get_memory_usage, detect_device, get_platform_hotkey, print_system_info


class HoldTranscribeApp:
    """Main HoldTranscribe application class."""

    def __init__(self):
        self.args: Optional[argparse.Namespace] = None
        self.device: Optional[str] = None
        self.audio_recorder: Optional[AudioRecorder] = None

        # Model references
        self.transcription_model: Optional[TranscriptionModel] = None
        self.assistant_model: Optional[AssistantModel] = None
        self.tts_model: Optional[TTSModel] = None

        # State tracking
        self.current_mode: Optional[str] = None
        self.transcription_start_time: Optional[float] = None

        # Audio playback configuration
        self.audio_buffer_size: int = 1024  # samples per buffer
        self.audio_queue_size: int = 10     # maximum queue size
        self.audio_timeout: float = 30.0    # playback timeout in seconds
        self.force_file_tts: bool = False   # force file-based TTS synthesis

        # Input handling (original implementation)
        self.transcribe_hotkey: Optional[set] = None
        self.assistant_hotkey: Optional[set] = None
        self.pressed: set = set()

        # Statistics
        self.model_load_time: float = 0

    def parse_arguments(self) -> argparse.Namespace:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description="Hold hotkey to record → transcribe → copy to clipboard",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s                          # Use defaults (large-v3 model, auto-detect device)
  %(prog)s --fast                   # Use base model for faster processing
  %(prog)s --model large-v2         # Use specific Whisper model
  %(prog)s --beam-size 1            # Faster transcription with beam size 1
  %(prog)s --debug                  # Enable detailed logging
  %(prog)s --tts                    # Enable text-to-speech for assistant responses
            """
        )

        # Model configuration
        parser.add_argument(
            "--model",
            default="mistralai/Voxtral-Mini-3B-2507",
            help="Model to use (default: %(default)s)"
        )
        parser.add_argument(
            "--fast",
            action="store_true",
            help="Use faster but less accurate transcription (base model)"
        )
        parser.add_argument(
            "--beam-size",
            type=int,
            default=5,
            help="Beam size for transcription (default: %(default)d)"
        )

        # TTS configuration
        parser.add_argument(
            "--tts",
            action="store_true",
            help="Enable text-to-speech for assistant responses"
        )
        parser.add_argument(
            "--tts-model",
            default="default",
            help="TTS model to use (default: ElevenLabs)"
        )
        parser.add_argument(
            "--tts-output",
            default="assistant_response.mp3",
            help="TTS output file pattern (default: %(default)s)"
        )
        parser.add_argument(
            "--force-file-tts",
            action="store_true",
            help="Force file-based TTS synthesis (disable streaming)"
        )

        # Audio playback configuration
        parser.add_argument(
            "--audio-buffer-size",
            type=int,
            default=1024,
            help="Audio buffer size in samples (default: %(default)s)"
        )
        parser.add_argument(
            "--audio-queue-size",
            type=int,
            default=10,
            help="Audio queue maximum size (default: %(default)s)"
        )
        parser.add_argument(
            "--audio-timeout",
            type=float,
            default=30.0,
            help="Audio playback timeout in seconds (default: %(default)s)"
        )

        # Debug and performance
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable debug output"
        )

        return parser.parse_args()

    def initialize_models(self) -> bool:
        """Initialize all required models."""
        if not self.args or not self.device:
            print("✗ ERROR: Configuration not initialized")
            return False

        debug_print("=== INITIALIZING MODELS ===")
        model_load_start = time.time()

        debug_print(f"Memory before model load: {get_memory_usage():.1f} MB")

        # Always load transcription model (Whisper)
        debug_print("Loading transcription model...")
        whisper_model_name = "base" if self.args.fast else "large-v3"

        self.transcription_model = ModelFactory.create_transcription_model(
            whisper_model_name,
            self.device,
            fast_mode=self.args.fast,
            beam_size=self.args.beam_size
        )

        if not self.transcription_model or not self.transcription_model.load():
            print("✗ ERROR: Failed to load transcription model")
            return False

        model_registry.register("transcription", self.transcription_model)

        # Load assistant model if specified
        if self.args.model.startswith("mistralai/Voxtral"):
            debug_print("Loading assistant model...")
            self.assistant_model = ModelFactory.create_assistant_model(
                self.args.model,
                self.device
            )

            if self.assistant_model and self.assistant_model.load():
                model_registry.register("assistant", self.assistant_model)
            else:
                debug_print("WARNING: Assistant model failed to load, assistant mode disabled")
                self.assistant_model = None

        # Load TTS model if enabled
        if self.args.tts:
            debug_print("Loading TTS model...")
            self.tts_model = ModelFactory.create_tts_model(
                self.args.tts_model,
                self.device
            )

            if self.tts_model and self.tts_model.load():
                model_registry.register("tts", self.tts_model)
            else:
                debug_print("WARNING: TTS model failed to load, TTS disabled")
                self.tts_model = None

        self.model_load_time = time.time() - model_load_start
        debug_print(f"Models loaded in {self.model_load_time:.3f}s")
        debug_print(f"Memory after model load: {get_memory_usage():.1f} MB")

        # Print summary
        models_loaded = ["Whisper"]
        if self.assistant_model:
            models_loaded.append("Voxtral")
        if self.tts_model:
            models_loaded.append("TTS")

        device_info = f"{self.device} {'(GPU accelerated)' if self.device == 'cuda' else '(CPU only)'}"
        speed_info = " (FAST MODE)" if self.args.fast else ""

        print(f"Models loaded ({' + '.join(models_loaded)}) in {self.model_load_time:.1f}s on {device_info}{speed_info}")

        if self.tts_model:
            print("🔊 Text-to-speech enabled for assistant responses")

        return True

    def setup_audio(self) -> bool:
        """Initialize audio recording system."""
        debug_print("=== INITIALIZING AUDIO ===")

        try:
            self.audio_recorder = AudioRecorder(
                sample_rate=16000,
                frame_duration_ms=30,
                vad_aggressiveness=2,
                channels=1
            )

            debug_print("Audio system initialized successfully")
            return True

        except Exception as e:
            print(f"✗ ERROR: Failed to initialize audio system: {e}")
            return False

    def setup_input_handling(self) -> bool:
        """Setup input handling and hotkeys."""
        debug_print("=== INITIALIZING INPUT HANDLING ===")

        try:
            # Get platform-specific hotkeys (original format with sets)
            transcribe_hotkey, assistant_hotkey, transcribe_msg, assistant_msg = get_platform_hotkey()

            # Store hotkeys and messages for the original input handling
            self.transcribe_hotkey = transcribe_hotkey
            self.assistant_hotkey = assistant_hotkey

            # Initialize pressed keys set for original hotkey detection
            self.pressed = set()

            # Print usage instructions
            print(f"{transcribe_msg}")
            if self.assistant_model:
                print(f"{assistant_msg}")
            print("(Run in background or add to your DE autostart.)")

            debug_print(f"Transcribe hotkey: {transcribe_hotkey}")
            debug_print(f"Assistant hotkey: {assistant_hotkey}")

            return True

        except Exception as e:
            print(f"✗ ERROR: Failed to setup input handling: {e}")
            return False

    def on_key_press(self, key):
        """Handle key press event (original implementation)."""
        key_time = time.time()
        debug_print(f"Key pressed: {key} at {key_time}")

        if key in self.transcribe_hotkey or key in self.assistant_hotkey:
            self.pressed.add(key)
            debug_print(f"Hotkey component pressed. Current pressed set: {self.pressed}")

        if self.pressed == self.transcribe_hotkey and not self.audio_recorder.is_recording_active():
            self.current_mode = "transcribe"
            debug_print("=== TRANSCRIBE HOTKEY COMBINATION COMPLETE ===")
            self.start_recording()
        elif self.pressed == self.assistant_hotkey and not self.audio_recorder.is_recording_active():
            self.current_mode = "assistant"
            debug_print("=== ASSISTANT HOTKEY COMBINATION COMPLETE ===")
            self.start_recording()

    def on_key_release(self, key):
        """Handle key release event (original implementation)."""
        key_time = time.time()
        debug_print(f"Key released: {key} at {key_time}")

        if key in self.pressed:
            self.pressed.discard(key)
            debug_print(f"Hotkey component released. Current pressed set: {self.pressed}")

        if self.audio_recorder.is_recording_active() and not self.transcribe_hotkey.issubset(self.pressed) and not self.assistant_hotkey.issubset(self.pressed):
            debug_print("=== HOTKEY COMBINATION RELEASED ===")
            self.stop_recording()

    def on_mouse_click(self, x, y, button, pressed_state):
        """Handle mouse click event (original implementation)."""
        mouse_time = time.time()
        debug_print(f"Mouse button {button} {'pressed' if pressed_state else 'released'} at ({x}, {y}) time: {mouse_time}")

        if button in self.transcribe_hotkey or button in self.assistant_hotkey:
            if pressed_state:
                self.pressed.add(button)
                debug_print(f"Mouse hotkey component pressed. Current pressed set: {self.pressed}")
                if self.pressed == self.transcribe_hotkey and not self.audio_recorder.is_recording_active():
                    self.current_mode = "transcribe"
                    debug_print("=== TRANSCRIBE MOUSE HOTKEY COMBINATION COMPLETE ===")
                    self.start_recording()
                elif self.pressed == self.assistant_hotkey and not self.audio_recorder.is_recording_active():
                    self.current_mode = "assistant"
                    debug_print("=== ASSISTANT MOUSE HOTKEY COMBINATION COMPLETE ===")
                    self.start_recording()
            else:
                self.pressed.discard(button)
                debug_print(f"Mouse hotkey component released. Current pressed set: {self.pressed}")
                if self.audio_recorder.is_recording_active() and not self.transcribe_hotkey.issubset(self.pressed) and not self.assistant_hotkey.issubset(self.pressed):
                    debug_print("=== MOUSE HOTKEY COMBINATION RELEASED ===")
                    self.stop_recording()

    def start_recording(self):
        """Start audio recording."""
        if not self.audio_recorder.start_recording():
            print("✗ Failed to start audio recording")
            return

        mode_msg = "🎤 Recording..." if self.current_mode == "transcribe" else "🤖 Recording for AI assistant..."
        print(f"{mode_msg} (release hotkey to process)")

    def stop_recording(self):
        """Stop audio recording and process."""
        print("🔄 Processing...")

        # Stop recording and get frames
        frames, stats = self.audio_recorder.stop_recording()

        if not frames:
            print("⚠️ No audio recorded")
            return

        # Process in background thread
        import threading
        if self.current_mode == "transcribe":
            threading.Thread(
                target=self._process_transcription,
                args=(frames, stats),
                daemon=True
            ).start()
        elif self.current_mode == "assistant":
            threading.Thread(
                target=self._process_assistant_request,
                args=(frames, stats),
                daemon=True
            ).start()

    def _process_transcription(self, frames: list, stats: dict):
        """Process transcription in background thread."""
        try:
            self.transcription_start_time = time.time()

            # Create temporary audio file
            temp_file = AudioUtils.create_temp_wav_file(frames, 16000)
            if not temp_file:
                print("✗ Failed to create temporary audio file")
                return

            try:
                # Transcribe audio
                if not self.transcription_model:
                    print("✗ Transcription model not available")
                    return

                transcription = self.transcription_model.transcribe(temp_file)

                if transcription.strip():
                    # Copy to clipboard
                    pyperclip.copy(transcription)

                    # Calculate timing
                    transcription_time = time.time() - self.transcription_start_time
                    audio_duration = stats.get('duration_seconds', 0)

                    print(f"✅ Transcribed ({transcription_time:.1f}s): {transcription}")
                    debug_print(f"Audio duration: {audio_duration:.1f}s, Processing time: {transcription_time:.1f}s")
                    debug_print(f"Speech ratio: {stats.get('speech_ratio', 0):.2f}")

                    # Send notification
                    try:
                        notify2.init("HoldTranscribe")
                        notice = notify2.Notification("Transcription Complete", transcription[:100] + ("..." if len(transcription) > 100 else ""))
                        notice.show()
                    except:
                        pass  # Notifications are optional

                else:
                    print("⚠️ No speech detected or transcription empty")

            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file)
                except:
                    pass

        except Exception as e:
            print(f"✗ Transcription failed: {e}")
            debug_print(f"Transcription error details: {e}")

        finally:
            # Reset state for next recording
            pass

    def _process_assistant_request(self, frames: list, stats: dict):
        """Process assistant request in background thread."""
        try:
            if not self.assistant_model:
                print("✗ Assistant model not available")
                return

            self.transcription_start_time = time.time()

            # Create temporary audio file
            temp_file = AudioUtils.create_temp_wav_file(frames, 16000)
            if not temp_file:
                print("✗ Failed to create temporary audio file")
                return

            try:
                # Generate assistant response
                response = self.assistant_model.generate_response(
                    temp_file,
                    prompt="Please provide a helpful response to the user's audio input."
                )

                if response.strip():
                    # Copy to clipboard
                    pyperclip.copy(response)

                    # Calculate timing
                    processing_time = time.time() - self.transcription_start_time
                    audio_duration = stats.get('duration_seconds', 0)

                    print(f"🤖 Assistant response ({processing_time:.1f}s): {response}")
                    debug_print(f"Audio duration: {audio_duration:.1f}s, Processing time: {processing_time:.1f}s")

                    # Generate TTS if enabled
                    if self.tts_model:
                        self._generate_tts(response)

                    # Send notification
                    try:
                        notify2.init("HoldTranscribe")
                        notice = notify2.Notification("AI Assistant Response", response[:100] + ("..." if len(response) > 100 else ""))
                        notice.show()
                    except:
                        pass  # Notifications are optional

                else:
                    print("⚠️ Assistant generated empty response")

            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file)
                except:
                    pass

        except Exception as e:
            print(f"✗ Assistant processing failed: {e}")
            debug_print(f"Assistant error details: {e}")

        finally:
            # Reset state for next recording
            pass

    def _generate_tts(self, text: str):
        """Generate text-to-speech for assistant response."""
        try:
            debug_print(f"_generate_tts called with text: '{text[:50]}...'")

            if not self.tts_model:
                debug_print("ERROR: No TTS model available in _generate_tts")
                return

            debug_print(f"TTS model available: {type(self.tts_model).__name__}")
            debug_print(f"TTS model loaded: {getattr(self.tts_model, 'is_loaded', 'Unknown')}")

            # Check if file-based TTS is forced
            if self.force_file_tts:
                debug_print("File-based TTS forced via --force-file-tts")
                self._generate_tts_file(text)
                return

            # Check if the TTS model supports streaming
            try:
                # Safely attempt streaming synthesis
                streaming_method = getattr(self.tts_model, 'synthesize_streaming', None)
                if streaming_method and callable(streaming_method):
                    debug_print("Using streaming TTS playback")
                    # Try multiple formats in order of preference
                    formats_to_try = ["pcm_22050", "pcm_16000", "mp3_44100_128"]
                    audio_stream = None

                    for format_attempt in formats_to_try:
                        try:
                            debug_print(f"Trying streaming format: {format_attempt}")
                            audio_stream = streaming_method(text, output_format=format_attempt)
                            if audio_stream:
                                debug_print(f"Successfully got stream with format: {format_attempt}")
                                break
                        except Exception as format_error:
                            debug_print(f"Format {format_attempt} failed: {str(format_error)[:200]}")
                            continue

                    if audio_stream:
                        streaming_success = self._play_audio_stream(audio_stream, text)
                        if not streaming_success:
                            debug_print("Streaming playback failed, falling back to file-based")
                            self._generate_tts_file(text)
                    else:
                        debug_print("All streaming formats failed, falling back to file-based")
                        self._generate_tts_file(text)
                else:
                    # Fallback to file-based synthesis
                    debug_print("TTS model doesn't support streaming, using file-based")
                    self._generate_tts_file(text)
            except (AttributeError, TypeError, Exception) as e:
                debug_print(f"Streaming TTS error: {e}, falling back to file-based")
                self._generate_tts_file(text)

        except Exception as e:
            debug_print(f"TTS error: {e}")
            import traceback
            debug_print(f"TTS traceback: {traceback.format_exc()}")

    def _generate_tts_file(self, text: str):
        """Generate TTS using file-based synthesis."""
        try:
            debug_print(f"_generate_tts_file called with text: '{text[:50]}...'")

            # Check if TTS model is available and loaded
            if not self.tts_model:
                debug_print("ERROR: No TTS model available")
                return

            if not hasattr(self.tts_model, 'is_loaded') or not self.tts_model.is_loaded:
                debug_print("ERROR: TTS model not loaded")
                return

            debug_print(f"TTS model info: {self.tts_model}")

            # Generate unique filename
            import uuid
            import os
            output_file = f"assistant_response_{uuid.uuid4().hex[:8]}.mp3"
            full_path = os.path.abspath(output_file)

            debug_print(f"Generating TTS audio file: {output_file}")
            debug_print(f"Full path: {full_path}")
            debug_print(f"Current working directory: {os.getcwd()}")

            # Call TTS synthesis
            debug_print("Calling TTS model synthesize method...")
            synthesis_result = self.tts_model.synthesize(text, output_file)
            debug_print(f"TTS synthesis result: {synthesis_result}")

            if synthesis_result:
                # Check if file was actually created
                if os.path.exists(output_file):
                    file_size = os.path.getsize(output_file)
                    debug_print(f"TTS audio file created: {output_file} ({file_size} bytes)")
                    self._play_audio_file(output_file)
                else:
                    debug_print(f"ERROR: TTS file was not created: {output_file}")
            else:
                debug_print("TTS synthesis returned False - generation failed")

        except Exception as e:
            debug_print(f"TTS file generation error: {e}")
            import traceback
            debug_print(f"Traceback: {traceback.format_exc()}")

    def _play_audio_stream(self, audio_stream, fallback_text=""):
        """Play streaming audio directly to the audio device with proper buffering."""
        debug_print("Starting streaming audio playback...")

        try:
            import sounddevice as sd
            import numpy as np
            import io
            import threading
            import queue
            import time

            try:
                from pydub import AudioSegment
                debug_print("pydub imported for streaming playback")
            except ImportError:
                debug_print("pydub not available for streaming playback, falling back")
                return

            # Audio configuration
            sample_rate = 22050  # ElevenLabs default (will be updated)
            channels = 1  # mono (will be updated)
            chunk_queue = queue.Queue(maxsize=self.audio_queue_size)
            playback_finished = threading.Event()
            stream_ended = threading.Event()
            audio_format_detected = threading.Event()
            playback_started = threading.Event()
            min_preload_chunks = max(3, self.audio_queue_size // 3)  # Preload some chunks

            def audio_callback(outdata, frames, time, status):
                """Callback function for sounddevice OutputStream."""
                if status:
                    debug_print(f"Audio callback status: {status}")

                try:
                    # Monitor queue level for debugging
                    queue_size = chunk_queue.qsize()
                    if queue_size < 2:  # Low buffer warning
                        debug_print(f"Low audio buffer: {queue_size} chunks remaining")

                    # Get audio data from queue
                    chunk_data = chunk_queue.get_nowait()
                    if chunk_data is None:  # End of stream signal
                        debug_print("Audio stream ended")
                        playback_finished.set()
                        raise sd.CallbackStop

                    # Ensure data fits the buffer
                    if len(chunk_data) < frames:
                        # Pad with zeros if chunk is shorter than buffer
                        padding = np.zeros(frames - len(chunk_data), dtype=np.float32)
                        chunk_data = np.concatenate([chunk_data, padding])
                    elif len(chunk_data) > frames:
                        # Take only what we need and put the rest back
                        remainder = chunk_data[frames:]
                        chunk_data = chunk_data[:frames]
                        # Put remainder back in queue for next callback
                        try:
                            chunk_queue.put_nowait(remainder)
                        except queue.Full:
                            debug_print("Audio queue full, dropping remainder")

                    if outdata.shape[1] == 1:  # Mono output
                        outdata[:, 0] = chunk_data[:frames]
                    else:  # Stereo output, duplicate mono to both channels
                        outdata[:, 0] = chunk_data[:frames]
                        outdata[:, 1] = chunk_data[:frames]

                except queue.Empty:
                    # No data available, output silence to prevent gaps
                    outdata.fill(0)
                    if playback_started.is_set():
                        debug_print("Audio underrun - no data available")
                except Exception as callback_error:
                    debug_print(f"Audio callback error: {callback_error}")
                    import traceback
                    debug_print(f"Callback traceback: {traceback.format_exc()}")
                    outdata.fill(0)

            def stream_processor():
                """Process streaming audio chunks and queue them for playback."""
                nonlocal sample_rate, channels
                try:
                    debug_print("Starting stream processor...")
                    chunk_count = 0
                    audio_buffer = b''  # Buffer for accumulating MP3 data
                    min_buffer_size = 4096  # Smaller buffer for lower latency
                    audio_initialized = False
                    decode_failures = 0
                    max_decode_failures = 5

                    for audio_chunk in audio_stream:
                        if stream_ended.is_set():
                            break

                        if audio_chunk:
                            chunk_count += 1
                            debug_print(f"Received audio chunk {chunk_count} ({len(audio_chunk)} bytes)")

                            # Add chunk to buffer
                            audio_buffer += audio_chunk

                            # Try to decode when we have enough data (more frequent attempts)
                            if len(audio_buffer) >= min_buffer_size or chunk_count % 5 == 0:
                                try:
                                    # Try PCM format first, then fall back to MP3
                                    try:
                                        # Assume PCM data: 16-bit, 44100Hz, mono
                                        audio_data = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32)
                                        audio_data = audio_data / 32768.0  # Normalize to [-1, 1]

                                        # Initialize audio parameters for PCM (detect from chunk size)
                                        if not audio_initialized:
                                            # Calculate sample rate based on chunk size assumption
                                            if len(audio_buffer) >= 4096:
                                                # Estimate: 16-bit samples, so bytes/2 = samples
                                                samples_per_chunk = len(audio_buffer) // 2
                                                # Common rates: 16000, 22050, 44100
                                                if samples_per_chunk > 1024:
                                                    sample_rate = 22050
                                                else:
                                                    sample_rate = 16000
                                            else:
                                                sample_rate = 22050  # Default
                                            channels = 1
                                            audio_initialized = True
                                            audio_format_detected.set()
                                            debug_print(f"Audio format initialized (PCM): {sample_rate}Hz, {channels}ch")

                                        # Split into playback chunks for smoother audio
                                        chunk_size = max(256, self.audio_buffer_size // 2)
                                        chunks_queued = 0
                                        for i in range(0, len(audio_data), chunk_size):
                                            sub_chunk = audio_data[i:i + chunk_size]
                                            chunk_queue.put(sub_chunk, timeout=10.0)
                                            chunks_queued += 1

                                        debug_print(f"Decoded {len(audio_data)} PCM samples from {len(audio_buffer)} bytes -> {chunks_queued} chunks")

                                    except (ValueError, TypeError):
                                        # Fallback to MP3 decoding
                                        audio_segment = AudioSegment.from_file(
                                            io.BytesIO(audio_buffer),
                                            format="mp3"
                                        )

                                        # Initialize audio parameters from first successful decode
                                        if not audio_initialized:
                                            sample_rate = audio_segment.frame_rate
                                            channels = audio_segment.channels
                                            audio_initialized = True
                                            audio_format_detected.set()  # Signal that format is known
                                            debug_print(f"Audio format initialized (MP3): {sample_rate}Hz, {channels}ch")

                                        # Convert to numpy array
                                        audio_data = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                                        audio_data = audio_data / np.iinfo(audio_segment.array_type).max

                                        # Handle stereo to mono conversion if needed
                                        if audio_segment.channels == 2:
                                            audio_data = audio_data.reshape((-1, 2))
                                            audio_data = np.mean(audio_data, axis=1)  # Convert to mono

                                        # Split into smaller playback chunks for smoother audio
                                        chunk_size = max(256, self.audio_buffer_size // 2)
                                        chunks_queued = 0
                                        for i in range(0, len(audio_data), chunk_size):
                                            sub_chunk = audio_data[i:i + chunk_size]
                                            chunk_queue.put(sub_chunk, timeout=10.0)
                                            chunks_queued += 1

                                        debug_print(f"Decoded {len(audio_data)} MP3 samples from {len(audio_buffer)} bytes -> {chunks_queued} chunks")

                                    # Signal when we have enough chunks to start playback
                                    if chunk_queue.qsize() >= min_preload_chunks and not playback_started.is_set():
                                        debug_print(f"Preload complete: {chunk_queue.qsize()} chunks ready")
                                        playback_started.set()

                                    # Clear buffer after successful decode
                                    audio_buffer = b''

                                except Exception as decode_error:
                                    # If decode fails, keep accumulating more data
                                    decode_failures += 1
                                    debug_print(f"Decode attempt failed ({decode_failures}/{max_decode_failures}) with {len(audio_buffer)} bytes: {str(decode_error)[:100]}")

                                    # If we have too many decode failures, abort streaming
                                    if decode_failures >= max_decode_failures:
                                        debug_print("Too many decode failures, aborting streaming")
                                        stream_ended.set()
                                        return False
                                    continue

                    # Process any remaining buffered data at end of stream
                    if audio_buffer:
                        try:
                            debug_print(f"Processing final buffer: {len(audio_buffer)} bytes")
                            # Try PCM first, then MP3
                            try:
                                audio_data = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32)
                                audio_data = audio_data / 32768.0
                                debug_print("Final buffer processed as PCM")
                            except (ValueError, TypeError):
                                audio_segment = AudioSegment.from_file(
                                    io.BytesIO(audio_buffer),
                                    format="mp3"
                                )
                                audio_data = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                                audio_data = audio_data / np.iinfo(audio_segment.array_type).max

                                if audio_segment.channels == 2:
                                    audio_data = audio_data.reshape((-1, 2))
                                    audio_data = np.mean(audio_data, axis=1)
                                debug_print("Final buffer processed as MP3")

                            chunk_size = self.audio_buffer_size
                            for i in range(0, len(audio_data), chunk_size):
                                sub_chunk = audio_data[i:i + chunk_size]
                                chunk_queue.put(sub_chunk, timeout=10.0)

                            debug_print("Final buffer processed successfully")
                            return True
                        except Exception as final_error:
                            debug_print(f"Failed to process final buffer: {final_error}")
                            return False

                    # Signal end of stream
                    debug_print(f"Stream processing complete. Processed {chunk_count} chunks.")
                    chunk_queue.put(None, timeout=10.0)
                    return True

                except Exception as stream_error:
                    debug_print(f"Stream processor error: {stream_error}")
                    import traceback
                    debug_print(f"Stream processor traceback: {traceback.format_exc()}")
                    chunk_queue.put(None)  # Signal end even on error
                    return False

            def play_stream():
                """Main streaming playback coordinator."""
                streaming_success = [False]  # Use list to allow modification from nested function

                def processor_wrapper():
                    streaming_success[0] = stream_processor()

                try:
                    # Start stream processor
                    processor_thread = threading.Thread(target=processor_wrapper)
                    processor_thread.daemon = True
                    processor_thread.start()

                    # Wait for audio format to be detected
                    debug_print("Waiting for audio format detection...")
                    if audio_format_detected.wait(timeout=10.0):
                        debug_print(f"Audio format detected: {sample_rate}Hz, {channels}ch")

                        # Wait for initial buffering before starting playback
                        debug_print(f"Waiting for preload ({min_preload_chunks} chunks)...")
                        if playback_started.wait(timeout=15.0):
                            debug_print(f"Starting audio stream: {sample_rate}Hz, {channels}ch")

                            # Create and start audio output stream with detected parameters
                            with sd.OutputStream(
                                samplerate=sample_rate,
                                channels=channels,
                                callback=audio_callback,
                                blocksize=max(256, self.audio_buffer_size // 2),
                                dtype=np.float32,
                                latency='low'
                            ):
                                # Wait for playback to complete
                                if playback_finished.wait(timeout=self.audio_timeout):
                                    debug_print("Streaming audio playback completed successfully")
                                else:
                                    debug_print("Streaming audio playback timed out")
                                    stream_ended.set()

                                # Wait for processor thread to complete
                                processor_thread.join(timeout=5.0)
                                return streaming_success[0]
                        else:
                            debug_print("Preload timeout - starting anyway")
                            stream_ended.set()
                            return False
                    else:
                        debug_print("Audio format detection timed out")
                        stream_ended.set()
                        return False

                except Exception as playback_error:
                    debug_print(f"Streaming playback error: {playback_error}")
                    import traceback
                    debug_print(f"Playback traceback: {traceback.format_exc()}")
                    return False

            # Start streaming playback and return success status
            debug_print("Starting streaming playback thread...")
            return play_stream()

        except ImportError as e:
            debug_print(f"Streaming audio requires additional dependencies: {e}")
            debug_print("Falling back to file-based synthesis")
            return False
        except Exception as e:
            debug_print(f"Streaming audio setup failed: {e}")
            import traceback
            debug_print(f"Streaming setup traceback: {traceback.format_exc()}")
            debug_print("Falling back to file-based synthesis")
            return False

    def _play_audio_file(self, audio_file: str):
        """Play audio file directly using sounddevice."""
        debug_print(f"_play_audio_file called with: {audio_file}")

        try:
            import sounddevice as sd
            import numpy as np
            import threading
            import os

            # Check file existence first
            if not os.path.exists(audio_file):
                debug_print(f"ERROR: Audio file does not exist: {audio_file}")
                debug_print(f"Current working directory: {os.getcwd()}")
                debug_print(f"Files in current directory: {os.listdir('.')}")
                return

            file_size = os.path.getsize(audio_file)
            debug_print(f"Audio file exists: {audio_file} ({file_size} bytes)")

            try:
                from pydub import AudioSegment
                debug_print("pydub imported successfully")
            except ImportError as pydub_error:
                debug_print(f"pydub not available: {pydub_error}")
                debug_print("falling back to system player")
                self._play_audio_system(audio_file)
                return

            def play_file():
                try:
                    debug_print(f"Thread: Loading audio file: {audio_file}")

                    # Load audio file using pydub
                    debug_print("Thread: Calling AudioSegment.from_file...")
                    audio = AudioSegment.from_file(audio_file)
                    debug_print(f"Thread: Audio loaded: {audio.frame_rate}Hz, {audio.channels}ch, {len(audio)}ms")

                    # Convert to numpy array
                    debug_print("Thread: Converting to numpy array...")
                    audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32)
                    debug_print(f"Thread: Original audio data range: {audio_data.min()} to {audio_data.max()}")

                    # Normalize to [-1, 1] range
                    audio_data = audio_data / np.iinfo(audio.array_type).max
                    debug_print(f"Thread: Normalized audio data range: {audio_data.min()} to {audio_data.max()}")

                    # Handle stereo/mono conversion
                    if audio.channels == 2:
                        audio_data = audio_data.reshape((-1, 2))

                    debug_print(f"Thread: Audio data shape: {audio_data.shape}")
                    debug_print("Thread: Starting sounddevice playback...")

                    # Use simple blocking playback for now
                    sd.play(audio_data, samplerate=audio.frame_rate, blocking=True)
                    debug_print("Thread: Audio playback completed successfully")

                    # Clean up the temporary file after playing
                    try:
                        os.remove(audio_file)
                        debug_print(f"Thread: Cleaned up temporary audio file: {audio_file}")
                    except Exception as cleanup_error:
                        debug_print(f"Thread: Warning: Could not clean up audio file {audio_file}: {cleanup_error}")

                except Exception as play_error:
                    debug_print(f"Thread: Direct audio playback failed: {play_error}")
                    import traceback
                    debug_print(f"Thread: Playback traceback: {traceback.format_exc()}")
                    # Fallback to system player
                    debug_print("Thread: Attempting fallback to system player...")
                    self._play_audio_system(audio_file)

            # Play audio in a separate thread to avoid blocking
            debug_print("Starting audio playback thread...")
            play_thread = threading.Thread(target=play_file)
            play_thread.daemon = True
            play_thread.start()

        except ImportError as e:
            debug_print(f"Import error: {e}")
            debug_print(f"Falling back to system player")
            self._play_audio_system(audio_file)
        except Exception as e:
            debug_print(f"Audio file playback setup failed: {e}")
            self._play_audio_system(audio_file)

    def _play_audio_system(self, audio_file: str):
        """Fallback audio playback using system player."""
        try:
            import platform
            import subprocess

            system = platform.system()
            if system == "Darwin":  # macOS
                subprocess.Popen(["afplay", audio_file])
            elif system == "Windows":
                subprocess.Popen(["start", audio_file], shell=True)
            else:  # Linux
                # Try multiple MP3 players in order of preference
                players = ["mpv", "vlc", "ffplay", "mplayer"]
                for player in players:
                    try:
                        subprocess.Popen([player, audio_file],
                                       stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)
                        debug_print(f"Playing audio with {player}: {audio_file}")
                        break
                    except FileNotFoundError:
                        continue
                else:
                    # Fallback to system default
                    subprocess.Popen(["xdg-open", audio_file])
                    debug_print(f"Playing audio with xdg-open: {audio_file}")

        except Exception as e:
            debug_print(f"System audio playback failed: {e}")

    def run(self) -> int:
        """Run the main application."""
        try:
            # Parse arguments
            self.args = self.parse_arguments()

            # Apply audio configuration from arguments
            self.audio_buffer_size = self.args.audio_buffer_size
            self.audio_queue_size = self.args.audio_queue_size
            self.audio_timeout = self.args.audio_timeout
            self.force_file_tts = self.args.force_file_tts

            # Setup debug mode
            set_debug(self.args.debug)

            if self.args.debug:
                print("[DEBUG MODE ENABLED - Extensive logging active]")

            # Print system information
            if self.args.debug:
                print_system_info()

            # Detect device
            self.device = detect_device()

            # Initialize components
            if not self.initialize_models():
                return 1

            if not self.setup_audio():
                return 1

            if not self.setup_input_handling():
                return 1

            debug_print("=== APPLICATION READY ===")
            debug_print("Waiting for hotkey input...")

            # Start input handling (original implementation)
            debug_print("Starting input listeners...")

            from pynput import keyboard, mouse

            try:
                with keyboard.Listener(on_press=self.on_key_press, on_release=self.on_key_release) as kb_listener, \
                     mouse.Listener(on_click=self.on_mouse_click) as mouse_listener:
                    debug_print("Input listeners started successfully")
                    debug_print("Application ready - waiting for hotkey input...")
                    kb_listener.join()
            except KeyboardInterrupt:
                debug_print("Application interrupted by user")

            return 0

        except Exception as e:
            print(f"✗ Application error: {e}")
            debug_print(f"Application error details: {e}")
            return 1

        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup application resources."""
        debug_print("=== APPLICATION CLEANUP ===")

        try:
            # Stop audio recording if active
            if self.audio_recorder and self.audio_recorder.is_recording_active():
                self.audio_recorder.stop_recording()

            # Unload all models
            model_registry.clear()

            debug_print("Application cleanup completed")

        except Exception as e:
            debug_print(f"Cleanup error: {e}")


def main():
    """Main entry point."""
    app = HoldTranscribeApp()
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
