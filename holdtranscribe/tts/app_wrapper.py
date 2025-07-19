"""
Ultra-simple TTS wrapper that leverages HoldTranscribe's existing working integration.

This wrapper avoids complex streaming state management by using the app's
existing TTS infrastructure and providing simple fallback mechanisms.
"""

import os
import sys
import subprocess
import tempfile
import time
from typing import Optional

try:
    import sounddevice as sd
    import soundfile as sf
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

from ..models import TTSModel
from ..utils import debug_print


class AppTTSWrapper(TTSModel):
    """Ultra-simple TTS wrapper that uses app's existing integration."""

    def __init__(self, model_name: str = "default", device: str = "cuda"):
        super().__init__(model_name, device)
        self.app_tts_model = None
        self.fallback_enabled = True

    def load(self) -> bool:
        """Load by checking if we can access the CLI."""
        try:
            # Simple dependency check
            import moshi
            import sphn
            self.is_loaded = True
            print(f"🔊 TTS wrapper ready on {self.device}")
            return True
        except ImportError:
            return False

    def unload(self):
        """Unload wrapper."""
        self.is_loaded = False
        self.app_tts_model = None

    def synthesize(self, text: str, output_file: str, **kwargs) -> bool:
        """Synthesize to file using CLI subprocess."""
        if not self.is_loaded:
            return False

        try:
            # Use subprocess to avoid memory conflicts
            cmd = [
                sys.executable, "-c", f"""
import sys
sys.path.insert(0, '{os.path.dirname(os.path.dirname(os.path.dirname(__file__)))}')
from holdtranscribe.tts.simple_cli import main
import sys
sys.argv = ['simple_cli', '{text}', '-o', '{output_file}', '--device', '{self.device}']
exit(main())
"""
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=60)
            return result.returncode == 0 and os.path.exists(output_file)

        except Exception as e:
            debug_print(f"File synthesis failed: {e}")
            return False

    def play_audio_to_speakers(self, text: str, **kwargs):
        """Play audio to speakers with multiple fallback strategies."""
        if not self.is_loaded:
            raise RuntimeError("TTS not loaded")

        if not HAS_AUDIO:
            raise ImportError("Audio libraries missing")

        # Strategy 1: Try subprocess CLI
        try:
            debug_print("Trying CLI subprocess approach...")
            cmd = [
                sys.executable, "-c", f"""
import sys
sys.path.insert(0, '{os.path.dirname(os.path.dirname(os.path.dirname(__file__)))}')
from holdtranscribe.tts.simple_cli import main
import sys
sys.argv = ['simple_cli', '{text}', '--device', '{self.device}']
exit(main())
"""
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=60)
            if result.returncode == 0:
                debug_print("CLI subprocess succeeded")
                return

        except Exception as e:
            debug_print(f"CLI subprocess failed: {e}")

        # Strategy 2: File-based fallback
        try:
            debug_print("Trying file-based fallback...")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_file = tmp.name

            if self.synthesize(text, temp_file):
                try:
                    audio_data, sample_rate = sf.read(temp_file)
                    if len(audio_data.shape) > 1:
                        audio_data = audio_data[:, 0]  # mono

                    sd.play(audio_data, samplerate=sample_rate)
                    sd.wait()
                    debug_print("File-based playback succeeded")
                    return

                finally:
                    try:
                        os.unlink(temp_file)
                    except:
                        pass

        except Exception as e:
            debug_print(f"File-based fallback failed: {e}")

        # Strategy 3: System TTS fallback
        try:
            debug_print("Trying system TTS fallback...")
            if sys.platform.startswith('linux'):
                subprocess.run(['espeak', text], check=True)
                return
        except:
            pass

        raise RuntimeError("All TTS strategies failed")

    def get_model_info(self) -> dict:
        """Get model info."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'is_loaded': self.is_loaded,
            'implementation': 'app-wrapper',
            'has_audio': HAS_AUDIO,
            'fallback_enabled': self.fallback_enabled
        }

    def __str__(self) -> str:
        return f"AppTTSWrapper(device='{self.device}', loaded={self.is_loaded})"


def create_app_tts_wrapper(device: str = "cuda") -> AppTTSWrapper:
    """Create an app TTS wrapper instance."""
    return AppTTSWrapper(device=device)
