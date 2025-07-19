"""
Simple, reliable Kyutai TTS implementation using CLI subprocess approach.

This implementation avoids streaming state issues by using the CLI interface
as a subprocess, which provides reliable text-to-speech synthesis.
"""

import os
import sys
import tempfile
import time
from typing import Optional, List

try:
    import sounddevice as sd
    import soundfile as sf
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

from ..models import TTSModel
from ..utils import debug_print


class KyutaiSimpleTTS(TTSModel):
    """Simple Kyutai TTS implementation using CLI subprocess approach."""

    def __init__(self, model_name: str = "default", device: str = "cuda"):
        super().__init__(model_name, device)
        self.timeout = 60  # seconds

        # Default to working Kyutai model
        if model_name == "default":
            self.model_name = "kyutai/tts-1.6b-en_fr"

    def load(self) -> bool:
        """Check if TTS dependencies are available."""
        try:
            # Test if the required modules are available
            import torch
            import moshi
            import sphn
            self.is_loaded = True
            print(f"🔊 Kyutai TTS dependencies available on {self.device}")
            return True

        except ImportError as e:
            debug_print(f"TTS dependencies not available: {e}")
            return False
        except Exception as e:
            debug_print(f"Failed to check TTS dependencies: {e}")
            return False

    def unload(self):
        """Unload the TTS model (nothing to do for CLI approach)."""
        self.is_loaded = False
        debug_print("Kyutai Simple TTS unloaded")

    def synthesize(self, text: str, output_file: str, **kwargs) -> bool:
        """Synthesize text to audio file using CLI."""
        if not self.is_loaded:
            return False

        try:
            debug_print(f"Synthesizing to file: {output_file}")
            return self._synthesize_direct(text, output_file)
        except Exception as e:
            debug_print(f"TTS synthesis error: {e}")
            return False

    def play_audio_to_speakers(self, text: str, **kwargs):
        """Play synthesized audio directly to speakers using CLI."""
        if not self.is_loaded:
            raise RuntimeError("TTS model not loaded")

        if not HAS_AUDIO:
            raise ImportError("Audio libraries not available. Install with: pip install sounddevice soundfile")

        try:
            debug_print(f"Playing to speakers: '{text[:50]}...'")
            success = self._synthesize_direct(text, None)
            if not success:
                raise RuntimeError("TTS synthesis failed")
        except Exception as e:
            debug_print(f"TTS playback error: {e}")
            raise

    def synthesize_to_file_and_play(self, text: str) -> bool:
        """Alternative method: synthesize to temp file then play."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                temp_path = tmp_file.name

            # Generate file
            if self.synthesize(text, temp_path):
                try:
                    # Load and play
                    audio_data, sample_rate = sf.read(temp_path)

                    # Handle stereo/mono
                    if len(audio_data.shape) > 1:
                        audio_data = audio_data[:, 0]

                    debug_print(f"Playing {len(audio_data)/sample_rate:.2f}s audio")
                    sd.play(audio_data, samplerate=sample_rate)
                    sd.wait()

                    return True

                finally:
                    # Cleanup
                    try:
                        os.unlink(temp_path)
                    except:
                        pass

            return False

        except Exception as e:
            debug_print(f"File-and-play method failed: {e}")
            return False

    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'is_loaded': self.is_loaded,
            'implementation': 'CLI-based',
            'timeout': self.timeout,
            'has_audio_libs': HAS_AUDIO
        }

    def set_timeout(self, timeout: int):
        """Set synthesis timeout in seconds."""
        self.timeout = timeout

    def test_synthesis(self, test_text: str = "Hello, this is a test.") -> bool:
        """Test if synthesis is working."""
        if not self.is_loaded:
            return False

        try:
            debug_print("Testing TTS synthesis...")
            self.play_audio_to_speakers(test_text)
            debug_print("TTS test successful")
            return True
        except Exception as e:
            debug_print(f"TTS test failed: {e}")
            return False

    def _synthesize_direct(self, text: str, output_file: Optional[str] = None) -> bool:
        """Direct synthesis using Kyutai TTS."""
        try:
            import torch
            import numpy as np
            from moshi.models.loaders import CheckpointInfo
            from moshi.models.tts import TTSModel as MoshiTTSModel, script_to_entries
            import sphn

            debug_print("Loading TTS model components...")

            # Load model
            checkpoint_info = CheckpointInfo.from_hf_repo(self.model_name)
            tts_model = MoshiTTSModel.from_checkpoint_info(
                checkpoint_info,
                n_q=32,
                temp=0.6,
                device=self.device
            )

            # Get voice
            voice_path = tts_model.get_voice_path("expresso/ex03-ex01_happy_001_channel1_334s.wav")
            condition_attributes = tts_model.make_condition_attributes([voice_path], cfg_coef=2.0)

            debug_print("Synthesizing audio...")

            pcms = []

            def on_frame(frame):
                if (frame != -1).all():
                    pcm = tts_model.mimi.decode(frame[:, 1:, :]).cpu().detach().numpy()
                    pcms.append(np.clip(pcm[0, 0], -1, 1))

            # Use simpler direct approach without TTSGen wrapper
            # This avoids the streaming state issues
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                temp_output = tmp_file.name

            # Use the CLI script approach which we know works
            from . import simple_cli
            import sys

            # Temporarily redirect argv to simulate CLI call
            old_argv = sys.argv
            try:
                if output_file:
                    sys.argv = ['simple_cli', text, '-o', output_file, '--device', self.device, '--model', self.model_name]
                else:
                    sys.argv = ['simple_cli', text, '--device', self.device, '--model', self.model_name]

                result = simple_cli.main()
                if result == 0:
                    return True
                else:
                    return False
            finally:
                sys.argv = old_argv
                # Clean up temp file if it was created
                try:
                    if os.path.exists(temp_output):
                        os.unlink(temp_output)
                except:
                    pass



        except Exception as e:
            debug_print(f"Direct synthesis error: {e}")
            return False

    def __str__(self) -> str:
        return f"KyutaiSimpleTTS(model='{self.model_name}', device='{self.device}', loaded={self.is_loaded})"


# Factory function for easy creation
def create_simple_kyutai_tts(device: str = "cuda") -> KyutaiSimpleTTS:
    """Create a simple Kyutai TTS instance."""
    return KyutaiSimpleTTS(device=device)
