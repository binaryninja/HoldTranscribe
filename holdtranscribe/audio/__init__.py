"""
Audio processing module for HoldTranscribe.

This module handles audio recording, voice activity detection (VAD),
and audio stream management.
"""

import threading
import time
import wave
import tempfile
from collections import deque
from typing import Optional, Callable, Any
import numpy as np
import sounddevice as sd
import webrtcvad

from ..utils import debug_print


class VADProcessor:
    """Voice Activity Detection processor using WebRTC VAD."""

    def __init__(self, aggressiveness: int = 2, sample_rate: int = 16000, frame_duration_ms: int = 30):
        """
        Initialize VAD processor.

        Args:
            aggressiveness: VAD aggressiveness level (0-3, 3 = most aggressive)
            sample_rate: Audio sample rate (must be 8000, 16000, 32000, or 48000)
            frame_duration_ms: Frame duration in milliseconds (10, 20, or 30)
        """
        if sample_rate not in [8000, 16000, 32000, 48000]:
            raise ValueError("Sample rate must be 8000, 16000, 32000, or 48000")

        if frame_duration_ms not in [10, 20, 30]:
            raise ValueError("Frame duration must be 10, 20, or 30 ms")

        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)

        debug_print(f"VAD initialized: aggressiveness={aggressiveness}, "
                   f"sample_rate={sample_rate}, frame_duration={frame_duration_ms}ms")

    def is_speech(self, frame: bytes) -> bool:
        """
        Determine if audio frame contains speech.

        Args:
            frame: Raw audio frame bytes

        Returns:
            True if frame contains speech, False otherwise
        """
        try:
            return self.vad.is_speech(frame, self.sample_rate)
        except Exception as e:
            debug_print(f"VAD error: {e}")
            return False

    def process_frames(self, frames: list) -> tuple[bool, int, int]:
        """
        Process multiple frames and return speech statistics.

        Args:
            frames: List of audio frame bytes

        Returns:
            Tuple of (has_speech, speech_frames, total_frames)
        """
        speech_frames = 0
        total_frames = len(frames)

        for frame in frames:
            if self.is_speech(frame):
                speech_frames += 1

        has_speech = speech_frames > 0
        return has_speech, speech_frames, total_frames


class AudioBuffer:
    """Thread-safe audio frame buffer."""

    def __init__(self, maxlen: Optional[int] = None):
        """
        Initialize audio buffer.

        Args:
            maxlen: Maximum number of frames to store (None = unlimited)
        """
        self.frames = deque(maxlen=maxlen)
        self.lock = threading.Lock()
        self.total_frames_added = 0

    def add_frame(self, frame: bytes):
        """Add a frame to the buffer."""
        with self.lock:
            self.frames.append(frame)
            self.total_frames_added += 1

    def get_frames(self, clear: bool = True) -> list:
        """
        Get all frames from buffer.

        Args:
            clear: Whether to clear the buffer after getting frames

        Returns:
            List of audio frame bytes
        """
        with self.lock:
            frames = list(self.frames)
            if clear:
                self.frames.clear()
            return frames

    def clear(self):
        """Clear all frames from buffer."""
        with self.lock:
            self.frames.clear()

    def size(self) -> int:
        """Get current number of frames in buffer."""
        with self.lock:
            return len(self.frames)

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        with self.lock:
            return len(self.frames) == 0


class AudioRecorder:
    """Audio recorder with VAD and streaming capabilities."""

    def __init__(self,
                 sample_rate: int = 16000,
                 frame_duration_ms: int = 30,
                 vad_aggressiveness: int = 2,
                 channels: int = 1):
        """
        Initialize audio recorder.

        Args:
            sample_rate: Audio sample rate
            frame_duration_ms: Frame duration for VAD processing
            vad_aggressiveness: VAD aggressiveness level (0-3)
            channels: Number of audio channels
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.channels = channels
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)

        # Initialize VAD processor
        self.vad_processor = VADProcessor(vad_aggressiveness, sample_rate, frame_duration_ms)

        # Initialize audio buffer
        self.buffer = AudioBuffer()

        # Stream management
        self.stream = None
        self.is_recording = False
        self.lock = threading.Lock()

        # Statistics
        self.start_time = None
        self.stop_time = None
        self.total_frames_processed = 0
        self.speech_frames_count = 0
        self.silence_frames_count = 0

        debug_print(f"AudioRecorder initialized: {sample_rate}Hz, {frame_duration_ms}ms frames, {channels} channel(s)")

    def _audio_callback(self, indata, frames, time, status):
        """Audio stream callback function."""
        if status:
            debug_print(f"Audio callback status: {status}")

        if not self.is_recording:
            return

        try:
            # Convert to bytes
            audio_bytes = (indata * 32767).astype(np.int16).tobytes()

            # Add to buffer
            self.buffer.add_frame(audio_bytes)

            # Update statistics
            self.total_frames_processed += 1

            # Check for speech using VAD
            if self.vad_processor.is_speech(audio_bytes):
                self.speech_frames_count += 1
            else:
                self.silence_frames_count += 1

        except Exception as e:
            debug_print(f"Audio callback error: {e}")

    def start_recording(self, device: Optional[int] = None) -> bool:
        """
        Start audio recording.

        Args:
            device: Audio input device ID (None = default)

        Returns:
            True if recording started successfully
        """
        with self.lock:
            if self.is_recording:
                debug_print("Recording already in progress")
                return True

            try:
                debug_print(f"Starting audio recording on device {device or 'default'}")

                # Reset statistics
                self.start_time = time.time()
                self.stop_time = None
                self.total_frames_processed = 0
                self.speech_frames_count = 0
                self.silence_frames_count = 0
                self.buffer.clear()

                # Start audio stream
                self.stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype=np.float32,
                    blocksize=self.frame_size,
                    device=device,
                    callback=self._audio_callback
                )

                self.stream.start()
                self.is_recording = True

                debug_print("Audio recording started successfully")
                return True

            except Exception as e:
                debug_print(f"Failed to start recording: {e}")
                self.stream = None
                self.is_recording = False
                return False

    def stop_recording(self) -> tuple[list, dict]:
        """
        Stop audio recording and return collected frames.

        Returns:
            Tuple of (frames_list, statistics_dict)
        """
        with self.lock:
            if not self.is_recording:
                debug_print("No recording in progress")
                return [], {}

            try:
                debug_print("Stopping audio recording")

                # Stop stream
                if self.stream is not None:
                    self.stream.stop()
                    self.stream.close()
                    self.stream = None

                self.is_recording = False
                self.stop_time = time.time()

                # Get collected frames
                frames = self.buffer.get_frames(clear=True)

                # Calculate statistics
                duration = self.stop_time - self.start_time if self.start_time else 0
                stats = {
                    'duration_seconds': duration,
                    'total_frames': len(frames),
                    'total_frames_processed': self.total_frames_processed,
                    'speech_frames': self.speech_frames_count,
                    'silence_frames': self.silence_frames_count,
                    'speech_ratio': self.speech_frames_count / max(1, self.total_frames_processed),
                    'sample_rate': self.sample_rate,
                    'frame_duration_ms': self.frame_duration_ms
                }

                debug_print(f"Recording stopped: {len(frames)} frames, {duration:.2f}s duration")
                debug_print(f"Speech ratio: {stats['speech_ratio']:.2f}")

                return frames, stats

            except Exception as e:
                debug_print(f"Error stopping recording: {e}")
                self.is_recording = False
                return [], {}

    def is_recording_active(self) -> bool:
        """Check if recording is currently active."""
        return self.is_recording

    def get_current_stats(self) -> dict:
        """Get current recording statistics."""
        current_time = time.time()
        duration = current_time - self.start_time if self.start_time else 0

        return {
            'is_recording': self.is_recording,
            'duration_seconds': duration,
            'total_frames_processed': self.total_frames_processed,
            'speech_frames': self.speech_frames_count,
            'silence_frames': self.silence_frames_count,
            'buffer_size': self.buffer.size(),
            'speech_ratio': self.speech_frames_count / max(1, self.total_frames_processed)
        }


class AudioUtils:
    """Utility functions for audio processing."""

    @staticmethod
    def frames_to_wav_file(frames: list, sample_rate: int, output_file: str) -> bool:
        """
        Save audio frames to WAV file.

        Args:
            frames: List of audio frame bytes
            sample_rate: Audio sample rate
            output_file: Output file path

        Returns:
            True if successful
        """
        try:
            with wave.open(output_file, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)

                # Write all frames
                for frame in frames:
                    wav_file.writeframes(frame)

            debug_print(f"Audio saved to: {output_file}")
            return True

        except Exception as e:
            debug_print(f"Failed to save audio: {e}")
            return False

    @staticmethod
    def frames_to_numpy(frames: list, sample_rate: int) -> np.ndarray:
        """
        Convert audio frames to numpy array.

        Args:
            frames: List of audio frame bytes
            sample_rate: Audio sample rate

        Returns:
            Numpy array of audio data
        """
        if not frames:
            return np.array([])

        # Concatenate all frames
        audio_bytes = b''.join(frames)

        # Convert to numpy array
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

        # Convert to float32 and normalize to [-1, 1]
        audio_array = audio_array.astype(np.float32) / 32767.0

        return audio_array

    @staticmethod
    def create_temp_wav_file(frames: list, sample_rate: int) -> Optional[str]:
        """
        Create a temporary WAV file from audio frames.

        Args:
            frames: List of audio frame bytes
            sample_rate: Audio sample rate

        Returns:
            Path to temporary WAV file
        """
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_file.close()

        if AudioUtils.frames_to_wav_file(frames, sample_rate, temp_file.name):
            return temp_file.name
        else:
            return None

    @staticmethod
    def get_audio_devices() -> list:
        """Get list of available audio devices."""
        try:
            devices = sd.query_devices()
            device_list = []

            for i, device in enumerate(devices):
                device_info = {
                    'id': i,
                    'name': device.name,
                    'hostapi': device.hostapi,
                    'max_input_channels': device.max_input_channels,
                    'max_output_channels': device.max_output_channels,
                    'default_samplerate': device.default_samplerate
                }
                device_list.append(device_info)

            return device_list

        except Exception as e:
            debug_print(f"Error querying audio devices: {e}")
            return []

    @staticmethod
    def test_audio_device(device_id: Optional[int] = None, duration: float = 1.0) -> bool:
        """
        Test if an audio device is working.

        Args:
            device_id: Device ID to test (None = default)
            duration: Test duration in seconds

        Returns:
            True if device is working
        """
        try:
            debug_print(f"Testing audio device {device_id or 'default'}")

            # Record for specified duration
            recording = sd.rec(
                int(duration * 16000),
                samplerate=16000,
                channels=1,
                device=device_id
            )
            sd.wait()

            # Check if we got any data
            if recording is not None and len(recording) > 0:
                debug_print("Audio device test successful")
                return True
            else:
                debug_print("Audio device test failed: no data recorded")
                return False

        except Exception as e:
            debug_print(f"Audio device test failed: {e}")
            return False


# Convenience function for quick recording
def quick_record(duration: float,
                sample_rate: int = 16000,
                device: Optional[int] = None) -> tuple[list, dict]:
    """
    Quickly record audio for a specified duration.

    Args:
        duration: Recording duration in seconds
        sample_rate: Audio sample rate
        device: Audio device ID

    Returns:
        Tuple of (frames, statistics)
    """
    recorder = AudioRecorder(sample_rate=sample_rate)

    if not recorder.start_recording(device=device):
        return [], {}

    time.sleep(duration)

    return recorder.stop_recording()
