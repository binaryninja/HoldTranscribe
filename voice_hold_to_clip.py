#!/usr/bin/env python3
"""
Hold the chosen hot‑key(s) to record → transcribe → copy to clipboard.

Requirements:
  pip install faster-whisper sounddevice pynput webrtcvad pyperclip notify2 numpy
"""

import os, sys, tempfile, threading, time, wave, subprocess, argparse, psutil
from collections import deque

import numpy as np
import sounddevice as sd
import webrtcvad
from pynput import keyboard, mouse
from faster_whisper import WhisperModel
import pyperclip, notify2

# ───────── CLI ARGUMENTS ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Voice-to-clipboard transcription with hold-to-record")
parser.add_argument("--debug", action="store_true", help="Enable extensive debugging output")
parser.add_argument("--fast", action="store_true", help="Use faster model and settings (lower accuracy)")
parser.add_argument("--model", default="large-v3", choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                    help="Whisper model size (default: large-v3)")
parser.add_argument("--beam-size", type=int, default=5, help="Beam size for transcription (1=fastest, 5=default)")
args = parser.parse_args()

DEBUG = args.debug
FAST_MODE = args.fast

def debug_print(*args_list, **kwargs):
    """Print debug messages only when DEBUG is enabled"""
    if DEBUG:
        timestamp = time.strftime('%H:%M:%S.%f')[:-3]  # millisecond precision
        print(f"[DEBUG {timestamp}]", *args_list, **kwargs)

def get_memory_usage():
    """Get current memory usage in MB"""
    if DEBUG:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    return 0

def detect_device():
    """Detect best available device (CUDA GPU or CPU)"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            debug_print(f"CUDA GPU detected: {gpu_name}")
            debug_print(f"CUDA version: {torch.version.cuda}")
            return "cuda"
    except ImportError:
        debug_print("PyTorch not available, CUDA detection skipped")
    except Exception as e:
        debug_print(f"CUDA detection failed: {e}")

    debug_print("Using CPU device")
    return "cpu"

# ───────── CONFIG ─────────────────────────────────────────────────────────────
HOTKEY = {keyboard.Key.ctrl, mouse.Button.button9}  # button9 = forward button (button8 = back)
debug_print(f"HOTKEY set to: {HOTKEY}")
SAMPLE_RATE = 16_000
FRAME_MS = 30                                   # 30‑ms frames for VAD

# Model selection based on arguments
if FAST_MODE:
    MODEL_NAME = "base"                         # Fast model for speed
    BEAM_SIZE = 1                               # Fastest beam search
    debug_print("FAST MODE: Using base model with beam_size=1")
else:
    MODEL_NAME = args.model                     # User-specified or default
    BEAM_SIZE = args.beam_size                  # User-specified or default

DEVICE = detect_device()                        # Auto-detect CUDA GPU or fallback to CPU

debug_print(f"Configuration:")
debug_print(f"  Sample Rate: {SAMPLE_RATE}")
debug_print(f"  Frame MS: {FRAME_MS}")
debug_print(f"  Model: {MODEL_NAME}")
debug_print(f"  Beam Size: {BEAM_SIZE}")
debug_print(f"  Fast Mode: {FAST_MODE}")
debug_print(f"  Device: {DEVICE} {'(GPU accelerated)' if DEVICE == 'cuda' else '(CPU only)'}")
debug_print(f"  Memory usage at startup: {get_memory_usage():.1f} MB")

# ───────── STATE VARIABLES ────────────────────────────────────────────────────
pressed = set()
vad = webrtcvad.Vad(2)                          # 0–3: 3 = most aggressive
frames = deque()                                # raw bytes
stream = None
model = None
lock = threading.Lock()

# Performance tracking
stream_start_time = None
stream_stop_time = None
transcription_start_time = None
model_load_time = None
total_frames_collected = 0
speech_frames_count = 0
silence_frames_count = 0

debug_print(f"VAD aggressiveness level: 2")
debug_print(f"Initial state initialized")

# ───────── AUDIO HELPERS ──────────────────────────────────────────────────────
def start_stream():
    global stream, frames, stream_start_time, total_frames_collected, speech_frames_count, silence_frames_count

    start_time = time.time()
    debug_print("=== STARTING AUDIO STREAM ===")
    debug_print(f"Memory before stream start: {get_memory_usage():.1f} MB")

    frames.clear()
    total_frames_collected = 0
    speech_frames_count = 0
    silence_frames_count = 0

    blocksize = int(SAMPLE_RATE * FRAME_MS / 1000)
    debug_print(f"Audio blocksize: {blocksize} samples ({FRAME_MS}ms)")

    try:
        stream = sd.RawInputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16',
                                   blocksize=blocksize,
                                   callback=callback)
        stream_creation_time = time.time() - start_time
        debug_print(f"Stream object created in {stream_creation_time*1000:.2f}ms")

        stream.start()
        stream_start_time = time.time()
        total_start_time = stream_start_time - start_time

        debug_print(f"Stream started successfully in {total_start_time*1000:.2f}ms total")
        debug_print(f"Memory after stream start: {get_memory_usage():.1f} MB")

    except Exception as e:
        debug_print(f"ERROR starting stream: {e}")
        stream = None

def stop_stream() -> bytes:
    """Stop recording, return raw audio data."""
    global stream, stream_stop_time

    stop_start_time = time.time()
    debug_print("=== STOPPING AUDIO STREAM ===")
    debug_print(f"Memory before stream stop: {get_memory_usage():.1f} MB")

    # Get stream reference and clear it first
    local_stream = stream
    stream = None
    stream_stop_time = time.time()

    # Stop stream outside of lock to prevent deadlock
    stream_stop_latency = 0
    if local_stream:
        try:
            stop_time = time.time()
            local_stream.stop()
            local_stream.close()
            stream_stop_latency = time.time() - stop_time
            debug_print(f"Stream stopped and closed in {stream_stop_latency*1000:.2f}ms")
        except Exception as e:
            debug_print(f"ERROR stopping stream: {e}")

    # Get frame data under lock
    frame_data_start = time.time()
    with lock:
        if not frames:
            debug_print("No frames collected, returning empty")
            return ""

        frame_count = len(frames)
        frame_data = b"".join(frames)
        data_size = len(frame_data)
        frames.clear()

    frame_data_time = time.time() - frame_data_start
    debug_print(f"Frame data collection: {frame_count} frames, {data_size} bytes in {frame_data_time*1000:.2f}ms")
    debug_print(f"Total frames: {total_frames_collected}, Speech: {speech_frames_count}, Silence: {silence_frames_count}")

    if stream_start_time:
        recording_duration = stream_stop_time - stream_start_time
        debug_print(f"Recording duration: {recording_duration:.3f}s")
        debug_print(f"Audio capture rate: {frame_count/recording_duration:.1f} frames/sec")

    # Return raw audio data instead of writing to file
    data_prep_time = time.time() - frame_data_start
    debug_print(f"Audio data prepared: {len(frame_data)} bytes in {data_prep_time*1000:.2f}ms")
    debug_print(f"Memory after data prep: {get_memory_usage():.1f} MB")

    total_stop_time = time.time() - stop_start_time
    debug_print(f"Total stream stop operation: {total_stop_time*1000:.2f}ms")

    return frame_data

def callback(indata, frames_cnt, time_info, status):
    # Called by PortAudio in a separate thread
    global total_frames_collected, speech_frames_count, silence_frames_count

    callback_start = time.time() if DEBUG else None

    if status and DEBUG:
        debug_print(f"Audio callback status: {status}")

    frame = bytes(indata)
    frame_size = len(frame)
    total_frames_collected += 1

    try:
        vad_start = time.time() if DEBUG else None
        is_speech = vad.is_speech(frame, SAMPLE_RATE)
        vad_time = (time.time() - vad_start) * 1000 if DEBUG else 0

        if is_speech:
            speech_frames_count += 1
            with lock:
                frames.append(frame)
            if DEBUG and total_frames_collected % 100 == 0:  # Log every 100 frames to avoid spam
                debug_print(f"Speech frame added (total speech: {speech_frames_count})")
        else:
            silence_frames_count += 1
            # keep small silence padding (300 ms) so we don't clip words
            with lock:
                if len(frames) < int(300 / FRAME_MS):
                    frames.append(frame)

        if DEBUG and callback_start:
            callback_time = (time.time() - callback_start) * 1000
            if total_frames_collected % 500 == 0:  # Log every 500 frames
                debug_print(f"Callback #{total_frames_collected}: {callback_time:.2f}ms total, VAD: {vad_time:.2f}ms, frame: {frame_size}B")

    except Exception as e:
        debug_print(f"ERROR in audio callback: {e}")

# ───────── KEYBOARD EVENT HANDLERS ────────────────────────────────────────────
def on_key_press(key):
    key_time = time.time()
    debug_print(f"Key pressed: {key} at {key_time}")

    if key in HOTKEY:
        pressed.add(key)
        debug_print(f"Hotkey component pressed. Current pressed set: {pressed}")

    if pressed == HOTKEY and stream is None:
        hotkey_complete_time = time.time()
        debug_print(f"=== HOTKEY COMBINATION COMPLETE ===")
        debug_print(f"Hotkey detection latency: {(hotkey_complete_time - key_time)*1000:.2f}ms")
        start_stream()

def on_key_release(key):
    key_time = time.time()
    debug_print(f"Key released: {key} at {key_time}")

    if key in pressed:
        pressed.discard(key)
        debug_print(f"Hotkey component released. Current pressed set: {pressed}")

    if stream and not HOTKEY.issubset(pressed):
        release_time = time.time()
        debug_print(f"=== HOTKEY COMBINATION RELEASED ===")
        debug_print(f"Release detection latency: {(release_time - key_time)*1000:.2f}ms")

        audio_data = stop_stream()
        if audio_data:
            debug_print(f"Starting transcription thread with {len(audio_data)} bytes")
            threading.Thread(target=transcribe_and_copy, args=(audio_data,), daemon=True).start()

# ───────── MOUSE EVENT HANDLERS ───────────────────────────────────────────────
def on_mouse_click(x, y, button, pressed_state):
    mouse_time = time.time()
    debug_print(f"Mouse button {button} {'pressed' if pressed_state else 'released'} at ({x}, {y}) time: {mouse_time}")

    if button in HOTKEY:
        if pressed_state:
            pressed.add(button)
            debug_print(f"Mouse hotkey component pressed. Current pressed set: {pressed}")
            if pressed == HOTKEY and stream is None:
                debug_print(f"=== MOUSE HOTKEY COMBINATION COMPLETE ===")
                start_stream()
        else:
            pressed.discard(button)
            debug_print(f"Mouse hotkey component released. Current pressed set: {pressed}")
            if stream and not HOTKEY.issubset(pressed):
                debug_print(f"=== MOUSE HOTKEY COMBINATION RELEASED ===")
                audio_data = stop_stream()
                if audio_data:
                    debug_print(f"Starting transcription thread with {len(audio_data)} bytes")
                    threading.Thread(target=transcribe_and_copy, args=(audio_data,), daemon=True).start()

# ───────── TRANSCRIPTION + CLIPBOARD ──────────────────────────────────────────
def transcribe_and_copy(audio_data: bytes):
    if not audio_data:
        debug_print("No audio data provided, skipping transcription")
        return

    global model, transcription_start_time

    transcription_start_time = time.time()
    debug_print(f"=== STARTING TRANSCRIPTION ===")
    debug_print(f"Audio data size: {len(audio_data)} bytes")
    debug_print(f"Memory before transcription: {get_memory_usage():.1f} MB")

    # Model should already be loaded at startup
    if model is None:
        debug_print("ERROR: Model not loaded! This should not happen.")
        return

    debug_print("Using pre-loaded model instance")

    # Create temporary WAV file in memory for Whisper
    wav_prep_start = time.time()
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
            with wave.open(tmp, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_data)

        wav_prep_time = time.time() - wav_prep_start
        debug_print(f"WAV file prepared in {wav_prep_time*1000:.2f}ms")
    except Exception as e:
        debug_print(f"ERROR preparing WAV file: {e}")
        return

    # Transcription timing with optimized parameters
    transcribe_start = time.time()
    try:
        debug_print(f"Starting Whisper transcription with beam_size={BEAM_SIZE}...")

        # Optimized transcription parameters
        transcribe_options = {
            "beam_size": BEAM_SIZE,
            "best_of": 1 if FAST_MODE else 5,  # Reduce candidates in fast mode
            "temperature": 0.0,  # Deterministic output
        }

        # Add fast mode optimizations
        if FAST_MODE:
            transcribe_options.update({
                "no_speech_threshold": 0.6,  # Higher threshold to skip silence faster
                "compression_ratio_threshold": 2.4,  # Skip low-quality audio faster
                "condition_on_previous_text": False,  # Disable context for speed
            })

        debug_print(f"Transcription options: {transcribe_options}")
        segments, info = model.transcribe(wav_path, **transcribe_options)

        text_segments = []
        segment_count = 0
        for segment in segments:
            text_segments.append(segment.text.strip())
            segment_count += 1
            if DEBUG and segment_count <= 5:  # Log first few segments
                debug_print(f"Segment {segment_count}: '{segment.text.strip()}'")

        text = " ".join(text_segments).strip()
        transcribe_time = time.time() - transcribe_start

        debug_print(f"Transcription completed in {transcribe_time:.3f}s")
        debug_print(f"Segments processed: {segment_count}")
        debug_print(f"Text length: {len(text)} characters")
        debug_print(f"Transcribed text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        debug_print(f"Memory after transcription: {get_memory_usage():.1f} MB")

    except Exception as e:
        debug_print(f"ERROR during transcription: {e}")
        text = ""
    finally:
        # Always cleanup the temporary file
        try:
            if 'wav_path' in locals():
                os.remove(wav_path)
                debug_print(f"WAV file cleaned up")
        except Exception as e:
            debug_print(f"ERROR removing WAV file: {e}")

    if not text:
        debug_print("No text to copy, skipping clipboard operation")
        return

    # Clipboard timing
    clipboard_start = time.time()
    try:
        pyperclip.copy(text)
        clipboard_time = time.time() - clipboard_start
        debug_print(f"Text copied to clipboard in {clipboard_time*1000:.2f}ms")
    except Exception as e:
        debug_print(f"ERROR copying to clipboard: {e}")

    # Notification timing
    notification_start = time.time()
    try:
        notify2.init("Transcriber")
        truncated_text = text[:120] + ("…" if len(text) > 120 else "")
        notify2.Notification("📝 Copied from speech", truncated_text).show()
        notification_time = time.time() - notification_start
        debug_print(f"Notification shown in {notification_time*1000:.2f}ms")
    except Exception as e:
        debug_print(f"ERROR showing notification: {e}")

    # Total timing analysis
    total_time = time.time() - transcription_start_time
    debug_print(f"=== TRANSCRIPTION COMPLETE ===")
    debug_print(f"Total transcription pipeline: {total_time:.3f}s")
    debug_print(f"Memory at completion: {get_memory_usage():.1f} MB")

    if stream_start_time and stream_stop_time:
        end_to_end_time = time.time() - stream_start_time
        recording_time = stream_stop_time - stream_start_time
        processing_time = total_time
        debug_print(f"=== END-TO-END TIMING ANALYSIS ===")
        debug_print(f"Recording duration: {recording_time:.3f}s")
        debug_print(f"Processing time: {processing_time:.3f}s")
        debug_print(f"Total end-to-end: {end_to_end_time:.3f}s")
        debug_print(f"Processing overhead: {(processing_time/recording_time)*100:.1f}% of recording time")

    print(time.strftime('%H:%M:%S'), "→", text)

# ───────── MAIN LOOP ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    startup_message = "Hold Ctrl+Mouse Forward Button to speak. Release to transcribe."
    if DEBUG:
        startup_message += "\n[DEBUG MODE ENABLED - Extensive logging active]"
    startup_message += "\n(Run in background or add to your DE autostart.)"

    print(startup_message)

    debug_print("=== APPLICATION STARTUP ===")
    debug_print(f"Python version: {sys.version}")
    debug_print(f"Process ID: {os.getpid()}")
    debug_print(f"Available audio devices:")
    if DEBUG:
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                debug_print(f"  {i}: {device['name']} ({device['hostapi']})")
        except Exception as e:
            debug_print(f"Error querying audio devices: {e}")

    # Pre-load Whisper model at startup
    debug_print("=== PRE-LOADING WHISPER MODEL ===")
    model_load_start = time.time()
    device_info = f"{DEVICE} {'(GPU accelerated)' if DEVICE == 'cuda' else '(CPU only)'}"
    debug_print(f"Loading Whisper model: {MODEL_NAME} on {device_info}")
    debug_print(f"Memory before model load: {get_memory_usage():.1f} MB")

    try:
        compute_type = "float16" if DEVICE == "cuda" else "int8"
        model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=compute_type)
        model_load_time = time.time() - model_load_start
        debug_print(f"Model pre-loaded successfully in {model_load_time:.3f}s using {compute_type}")
        debug_print(f"Memory after model load: {get_memory_usage():.1f} MB")
        device_emoji = "🚀" if DEVICE == "cuda" else "🖥️"
        speed_info = " (FAST MODE)" if FAST_MODE else ""
        print(f"{device_emoji} Whisper model '{MODEL_NAME}' loaded in {model_load_time:.1f}s on {device_info}{speed_info}")
    except Exception as e:
        print(f"✗ ERROR loading Whisper model: {e}")
        debug_print(f"FATAL ERROR loading model: {e}")
        sys.exit(1)

    debug_print("Starting input listeners...")

    try:
        with keyboard.Listener(on_press=on_key_press, on_release=on_key_release) as kb_listener, \
             mouse.Listener(on_click=on_mouse_click) as mouse_listener:
            debug_print("Input listeners started successfully")
            debug_print("Application ready - waiting for hotkey input...")
            kb_listener.join()
    except KeyboardInterrupt:
        debug_print("Application interrupted by user")
    except Exception as e:
        debug_print(f"ERROR in main loop: {e}")
    finally:
        debug_print("Application shutting down...")
