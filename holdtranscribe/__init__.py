"""
HoldTranscribe - Hotkey-Activated Voice-to-Clipboard Transcriber

A lightweight tool that records audio while you hold a configurable hotkey,
transcribes speech using OpenAI's Whisper model, and copies the result to your clipboard.
"""

__version__ = "1.0.1"
__author__ = "binaryninja"
__email__ = ""
__description__ = "Hotkey-Activated Voice-to-Clipboard Transcriber"
__url__ = "https://github.com/binaryninja/holdtranscribe"

from .main import main

__all__ = ["main", "__version__"]
