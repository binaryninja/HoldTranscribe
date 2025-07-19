"""
Utility functions for HoldTranscribe application.
"""

import os
import platform
import psutil
from pynput import keyboard, mouse


# Global debug flag - will be set by main application
DEBUG = False


def set_debug(enabled: bool):
    """Set global debug flag."""
    global DEBUG
    DEBUG = enabled


def debug_print(message: str):
    """Print debug message if debug mode is enabled."""
    if DEBUG:
        print(f"[DEBUG] {message}")


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def detect_device() -> str:
    """Detect best available device (CUDA GPU or CPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            debug_print(f"CUDA GPU detected: {gpu_name}")
            try:
                debug_print(f"CUDA version: {torch.version.cuda}")
            except AttributeError:
                debug_print("CUDA version: Unknown")
            return "cuda"
    except ImportError:
        debug_print("PyTorch not available, CUDA detection skipped")
    except Exception as e:
        debug_print(f"CUDA detection failed: {e}")

    debug_print("Using CPU device")
    return "cpu"


def get_platform_hotkey() -> tuple[set, set, str, str]:
    """Get platform-specific hotkey configuration."""
    system = platform.system().lower()

    if system == 'darwin':  # macOS
        transcribe_hotkey = {keyboard.Key.ctrl, keyboard.Key.alt, keyboard.Key.cmd, keyboard.Key.space}
        assistant_hotkey = {keyboard.Key.ctrl, keyboard.Key.shift, mouse.Button.button8}
        transcribe_message = "Hold Ctrl+Option+Cmd+Space to speak. Release to transcribe."
        assistant_message = "Hold Ctrl+Shift+Back Mouse Button to speak. Release for AI assistant."
    elif system == 'linux':
        transcribe_hotkey = {keyboard.Key.ctrl, mouse.Button.button9}
        assistant_hotkey = {keyboard.Key.ctrl, keyboard.Key.shift, mouse.Button.button8}
        transcribe_message = "Hold Ctrl+Mouse Forward Button to speak. Release to transcribe."
        assistant_message = "Hold Ctrl+Shift+Back Mouse Button to speak. Release for AI assistant."
    else:  # Windows or other
        transcribe_hotkey = {keyboard.Key.ctrl, keyboard.Key.space}
        assistant_hotkey = {keyboard.Key.ctrl, keyboard.Key.shift, mouse.Button.button8}
        transcribe_message = "Hold Ctrl+Space to speak. Release to transcribe."
        assistant_message = "Hold Ctrl+Shift+Back Mouse Button to speak. Release for AI assistant."

    return transcribe_hotkey, assistant_hotkey, transcribe_message, assistant_message


def print_system_info():
    """Print system information for debugging."""
    debug_print("=== SYSTEM INFORMATION ===")
    debug_print(f"Platform: {platform.system()} {platform.release()}")
    debug_print(f"Architecture: {platform.machine()}")
    debug_print(f"Python version: {platform.python_version()}")
    debug_print(f"Process ID: {os.getpid()}")
    debug_print(f"Memory usage at startup: {get_memory_usage():.1f} MB")
