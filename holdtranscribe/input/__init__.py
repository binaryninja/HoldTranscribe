"""
Input handling module for HoldTranscribe.

This module handles keyboard and mouse input events, including hotkey detection
and management of recording states.
"""

import threading
from typing import Callable, Optional, Set, Dict, Any
from enum import Enum
from pynput import keyboard, mouse

from ..utils import debug_print


class InputMode(Enum):
    """Input modes for different functionalities."""
    TRANSCRIBE = "transcribe"
    ASSISTANT = "assistant"


class HotkeyState(Enum):
    """States of hotkey interaction."""
    IDLE = "idle"
    PRESSED = "pressed"
    RECORDING = "recording"
    PROCESSING = "processing"


class HotkeyManager:
    """Manages hotkey combinations and their states."""

    def __init__(self):
        self.hotkeys = {}
        self.pressed_keys = set()
        self.current_state = HotkeyState.IDLE
        self.current_mode = None
        self.lock = threading.Lock()

    def register_hotkey(self,
                       mode: InputMode,
                       key_combination: str,
                       on_press: Optional[Callable] = None,
                       on_release: Optional[Callable] = None):
        """
        Register a hotkey combination.

        Args:
            mode: Input mode for this hotkey
            key_combination: String representation of key combination (e.g., "ctrl+shift+t")
            on_press: Callback function when hotkey is pressed
            on_release: Callback function when hotkey is released
        """
        keys = self._parse_key_combination(key_combination)

        self.hotkeys[mode] = {
            'keys': keys,
            'combination_string': key_combination,
            'on_press': on_press,
            'on_release': on_release
        }

        debug_print(f"Registered hotkey for {mode.value}: {key_combination}")

    def _parse_key_combination(self, combination: str) -> Set[str]:
        """Parse key combination string into set of keys."""
        keys = set()
        parts = combination.lower().split('+')

        for part in parts:
            part = part.strip()
            # Map common key names
            if part == 'ctrl':
                keys.add('ctrl_l')  # Default to left ctrl
            elif part == 'shift':
                keys.add('shift_l')  # Default to left shift
            elif part == 'alt':
                keys.add('alt_l')
            elif part == 'cmd':
                keys.add('cmd')
            else:
                keys.add(part)

        return keys

    def _normalize_key(self, key) -> str:
        """Normalize key object to string representation."""
        try:
            if hasattr(key, 'char') and key.char is not None:
                return key.char.lower()
            elif hasattr(key, 'name'):
                return key.name.lower()
            else:
                return str(key).lower()
        except AttributeError:
            return str(key).lower()

    def on_key_press(self, key) -> Optional[InputMode]:
        """
        Handle key press event.

        Args:
            key: Key object from pynput

        Returns:
            InputMode if a hotkey combination is activated, None otherwise
        """
        with self.lock:
            key_str = self._normalize_key(key)
            self.pressed_keys.add(key_str)

            debug_print(f"Key pressed: {key_str}, current keys: {self.pressed_keys}")

            # Check if any hotkey combination is satisfied
            for mode, hotkey_info in self.hotkeys.items():
                if hotkey_info['keys'].issubset(self.pressed_keys):
                    if self.current_state == HotkeyState.IDLE:
                        self.current_state = HotkeyState.PRESSED
                        self.current_mode = mode

                        debug_print(f"Hotkey activated: {mode.value}")

                        # Call press callback if registered
                        if hotkey_info['on_press']:
                            try:
                                hotkey_info['on_press'](mode)
                            except Exception as e:
                                debug_print(f"Error in hotkey press callback: {e}")

                        return mode

            return None

    def on_key_release(self, key) -> Optional[InputMode]:
        """
        Handle key release event.

        Args:
            key: Key object from pynput

        Returns:
            InputMode if a hotkey combination is deactivated, None otherwise
        """
        with self.lock:
            key_str = self._normalize_key(key)
            self.pressed_keys.discard(key_str)

            debug_print(f"Key released: {key_str}, current keys: {self.pressed_keys}")

            # Check if the current hotkey combination is no longer satisfied
            if self.current_mode is not None:
                current_hotkey = self.hotkeys[self.current_mode]
                if not current_hotkey['keys'].issubset(self.pressed_keys):
                    released_mode = self.current_mode

                    debug_print(f"Hotkey deactivated: {released_mode.value}")

                    # Call release callback if registered
                    if current_hotkey['on_release']:
                        try:
                            current_hotkey['on_release'](released_mode)
                        except Exception as e:
                            debug_print(f"Error in hotkey release callback: {e}")

                    self.current_state = HotkeyState.IDLE
                    self.current_mode = None

                    return released_mode

            return None

    def set_state(self, state: HotkeyState):
        """Set the current hotkey state."""
        with self.lock:
            self.current_state = state
            debug_print(f"Hotkey state changed to: {state.value}")

    def get_state(self) -> HotkeyState:
        """Get the current hotkey state."""
        return self.current_state

    def get_current_mode(self) -> Optional[InputMode]:
        """Get the currently active input mode."""
        return self.current_mode

    def is_hotkey_active(self, mode: Optional[InputMode] = None) -> bool:
        """Check if a hotkey is currently active."""
        if mode is None:
            return self.current_mode is not None
        else:
            return self.current_mode == mode

    def clear_pressed_keys(self):
        """Clear all pressed keys (useful for cleanup)."""
        with self.lock:
            self.pressed_keys.clear()
            self.current_state = HotkeyState.IDLE
            self.current_mode = None


class MouseManager:
    """Manages mouse events."""

    def __init__(self):
        self.click_callbacks = []
        self.move_callbacks = []
        self.scroll_callbacks = []

    def register_click_callback(self, callback: Callable):
        """Register a callback for mouse click events."""
        self.click_callbacks.append(callback)

    def register_move_callback(self, callback: Callable):
        """Register a callback for mouse move events."""
        self.move_callbacks.append(callback)

    def register_scroll_callback(self, callback: Callable):
        """Register a callback for mouse scroll events."""
        self.scroll_callbacks.append(callback)

    def on_click(self, x, y, button, pressed):
        """Handle mouse click events."""
        debug_print(f"Mouse {'pressed' if pressed else 'released'}: {button} at ({x}, {y})")

        for callback in self.click_callbacks:
            try:
                callback(x, y, button, pressed)
            except Exception as e:
                debug_print(f"Error in mouse click callback: {e}")

    def on_move(self, x, y):
        """Handle mouse move events."""
        for callback in self.move_callbacks:
            try:
                callback(x, y)
            except Exception as e:
                debug_print(f"Error in mouse move callback: {e}")

    def on_scroll(self, x, y, dx, dy):
        """Handle mouse scroll events."""
        debug_print(f"Mouse scroll at ({x}, {y}): ({dx}, {dy})")

        for callback in self.scroll_callbacks:
            try:
                callback(x, y, dx, dy)
            except Exception as e:
                debug_print(f"Error in mouse scroll callback: {e}")


class InputManager:
    """Main input manager that coordinates keyboard and mouse handling."""

    def __init__(self):
        self.hotkey_manager = HotkeyManager()
        self.mouse_manager = MouseManager()

        self.keyboard_listener = None
        self.mouse_listener = None
        self.is_listening = False

        self.stop_callbacks = []

    def register_hotkey(self,
                       mode: InputMode,
                       key_combination: str,
                       on_press: Optional[Callable] = None,
                       on_release: Optional[Callable] = None):
        """Register a hotkey combination."""
        self.hotkey_manager.register_hotkey(mode, key_combination, on_press, on_release)

    def register_mouse_callback(self, event_type: str, callback: Callable):
        """Register mouse event callback."""
        if event_type == 'click':
            self.mouse_manager.register_click_callback(callback)
        elif event_type == 'move':
            self.mouse_manager.register_move_callback(callback)
        elif event_type == 'scroll':
            self.mouse_manager.register_scroll_callback(callback)
        else:
            raise ValueError(f"Unknown mouse event type: {event_type}")

    def register_stop_callback(self, callback: Callable):
        """Register callback to be called when input stops."""
        self.stop_callbacks.append(callback)

    def start_listening(self) -> bool:
        """Start listening for input events."""
        if self.is_listening:
            debug_print("Input manager already listening")
            return True

        try:
            debug_print("Starting input listeners...")

            # Start keyboard listener
            self.keyboard_listener = keyboard.Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release
            )

            # Start mouse listener
            self.mouse_listener = mouse.Listener(
                on_click=self.mouse_manager.on_click,
                on_move=self.mouse_manager.on_move,
                on_scroll=self.mouse_manager.on_scroll
            )

            self.keyboard_listener.start()
            self.mouse_listener.start()

            self.is_listening = True
            debug_print("Input listeners started successfully")
            return True

        except Exception as e:
            debug_print(f"Failed to start input listeners: {e}")
            return False

    def stop_listening(self):
        """Stop listening for input events."""
        if not self.is_listening:
            return

        debug_print("Stopping input listeners...")

        try:
            if self.keyboard_listener:
                self.keyboard_listener.stop()
                self.keyboard_listener = None

            if self.mouse_listener:
                self.mouse_listener.stop()
                self.mouse_listener = None

            self.is_listening = False

            # Call stop callbacks
            for callback in self.stop_callbacks:
                try:
                    callback()
                except Exception as e:
                    debug_print(f"Error in stop callback: {e}")

            debug_print("Input listeners stopped")

        except Exception as e:
            debug_print(f"Error stopping input listeners: {e}")

    def _on_key_press(self, key):
        """Internal keyboard press handler."""
        try:
            self.hotkey_manager.on_key_press(key)
        except Exception as e:
            debug_print(f"Error handling key press: {e}")

    def _on_key_release(self, key):
        """Internal keyboard release handler."""
        try:
            self.hotkey_manager.on_key_release(key)
        except Exception as e:
            debug_print(f"Error handling key release: {e}")

    def wait_for_stop(self):
        """Wait for keyboard listener to stop (blocking)."""
        if self.keyboard_listener:
            self.keyboard_listener.join()

    def get_hotkey_state(self) -> HotkeyState:
        """Get current hotkey state."""
        return self.hotkey_manager.get_state()

    def set_hotkey_state(self, state: HotkeyState):
        """Set hotkey state."""
        self.hotkey_manager.set_state(state)

    def get_current_mode(self) -> Optional[InputMode]:
        """Get currently active input mode."""
        return self.hotkey_manager.get_current_mode()

    def is_hotkey_active(self, mode: Optional[InputMode] = None) -> bool:
        """Check if hotkey is active."""
        return self.hotkey_manager.is_hotkey_active(mode)

    def cleanup(self):
        """Cleanup resources."""
        self.stop_listening()
        self.hotkey_manager.clear_pressed_keys()


# Global input manager instance
input_manager = InputManager()


# Convenience functions
def setup_transcription_hotkey(key_combination: str, on_press: Optional[Callable], on_release: Optional[Callable]):
    """Setup hotkey for transcription mode."""
    input_manager.register_hotkey(InputMode.TRANSCRIBE, key_combination, on_press, on_release)


def setup_assistant_hotkey(key_combination: str, on_press: Optional[Callable], on_release: Optional[Callable]):
    """Setup hotkey for assistant mode."""
    input_manager.register_hotkey(InputMode.ASSISTANT, key_combination, on_press, on_release)


def start_input_handling() -> bool:
    """Start input event handling."""
    return input_manager.start_listening()


def stop_input_handling():
    """Stop input event handling."""
    input_manager.stop_listening()


def wait_for_input():
    """Wait for input events (blocking)."""
    input_manager.wait_for_stop()
