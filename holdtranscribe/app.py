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
            if not self.tts_model:
                return

            # Generate unique filename
            import uuid
            output_file = f"assistant_response_{uuid.uuid4().hex[:8]}.mp3"

            debug_print(f"Generating TTS audio: {output_file}")

            if self.tts_model.synthesize(text, output_file):
                debug_print(f"TTS audio generated: {output_file}")

                # Play the audio (platform-specific)
                self._play_audio(output_file)
            else:
                debug_print("TTS generation failed")

        except Exception as e:
            debug_print(f"TTS error: {e}")

    def _play_audio(self, audio_file: str):
        """Play audio file using system player."""
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
            debug_print(f"Audio playback failed: {e}")

    def run(self) -> int:
        """Run the main application."""
        try:
            # Parse arguments
            self.args = self.parse_arguments()

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
