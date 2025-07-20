#!/usr/bin/env python3
"""
Test script for audio playback functionality in HoldTranscribe.

This script tests the new direct audio device playback methods.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

# Add the HoldTranscribe package to path
sys.path.insert(0, str(Path(__file__).parent))

def test_dependencies():
    """Test if required dependencies are available."""
    print("Testing dependencies...")

    try:
        import sounddevice as sd
        print("✓ sounddevice available")
    except ImportError:
        print("✗ sounddevice not available - install with: pip install sounddevice")
        return False

    try:
        from pydub import AudioSegment
        print("✓ pydub available")
    except ImportError:
        print("✗ pydub not available - install with: pip install pydub")
        return False

    try:
        import numpy as np
        print("✓ numpy available")
    except ImportError:
        print("✗ numpy not available")
        return False

    return True

def create_test_audio():
    """Create a simple test audio file."""
    try:
        from pydub import AudioSegment
        from pydub.generators import Sine
        import tempfile

        # Generate a 2-second 440Hz tone (A4 note)
        tone = Sine(440).to_audio_segment(duration=2000)  # 2 seconds

        # Create temporary file
        temp_file = tempfile.mktemp(suffix='.mp3')
        tone.export(temp_file, format="mp3")

        print(f"✓ Created test audio file: {temp_file}")
        return temp_file
    except Exception as e:
        print(f"✗ Failed to create test audio: {e}")
        return None

def test_direct_audio_playback(audio_file):
    """Test direct audio playback using sounddevice."""
    print("\nTesting direct audio playback...")

    try:
        import sounddevice as sd
        import numpy as np
        from pydub import AudioSegment

        # Load audio file
        audio = AudioSegment.from_file(audio_file)
        print(f"  Audio format: {audio.frame_rate}Hz, {audio.channels} channels, {len(audio)}ms")

        # Convert to numpy array
        audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32)
        audio_data = audio_data / np.iinfo(audio.array_type).max

        # Handle stereo/mono
        if audio.channels == 2:
            audio_data = audio_data.reshape((-1, 2))

        print("  Playing audio...")
        sd.play(audio_data, samplerate=audio.frame_rate, blocking=True)
        print("✓ Direct audio playback successful")
        return True

    except Exception as e:
        print(f"✗ Direct audio playback failed: {e}")
        return False

def test_buffered_audio_playback(audio_file):
    """Test buffered audio playback using OutputStream."""
    print("\nTesting buffered audio playback...")

    try:
        import sounddevice as sd
        import numpy as np
        from pydub import AudioSegment
        import queue
        import threading
        import time

        # Load audio file
        audio = AudioSegment.from_file(audio_file)
        print(f"  Audio format: {audio.frame_rate}Hz, {audio.channels} channels, {len(audio)}ms")

        # Convert to numpy array
        audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32)
        audio_data = audio_data / np.iinfo(audio.array_type).max

        # Handle stereo/mono
        if audio.channels == 2:
            audio_data = audio_data.reshape((-1, 2))

        # Buffer configuration
        buffer_size = 1024
        audio_queue = queue.Queue(maxsize=10)
        playback_finished = threading.Event()
        start_time = time.time()

        def audio_callback(outdata, frames, time, status):
            """Callback function for sounddevice OutputStream."""
            if status:
                print(f"    Audio callback status: {status}")

            try:
                data = audio_queue.get_nowait()
                if data is None:  # End of stream
                    playback_finished.set()
                    raise sd.CallbackStop

                # Ensure data fits the buffer
                if len(data) < frames:
                    padding = np.zeros((frames - len(data), audio.channels), dtype=np.float32)
                    data = np.vstack([data, padding]) if len(data.shape) == 2 else np.concatenate([data, padding.flatten()])
                elif len(data) > frames:
                    data = data[:frames]

                outdata[:] = data.reshape(frames, audio.channels) if len(data.shape) == 1 else data

            except queue.Empty:
                outdata.fill(0)

        # Queue audio data
        def queue_audio():
            try:
                for i in range(0, len(audio_data), buffer_size):
                    chunk = audio_data[i:i + buffer_size]
                    audio_queue.put(chunk, timeout=5.0)
                audio_queue.put(None)  # End signal
            except Exception as e:
                print(f"    Error queuing audio: {e}")
                audio_queue.put(None)

        # Start queuing in separate thread
        queue_thread = threading.Thread(target=queue_audio)
        queue_thread.daemon = True
        queue_thread.start()

        print("  Playing buffered audio...")

        # Play with OutputStream
        with sd.OutputStream(
            samplerate=audio.frame_rate,
            channels=audio.channels,
            callback=audio_callback,
            blocksize=buffer_size,
            dtype=np.float32
        ):
            playback_finished.wait(timeout=30)

        end_time = time.time()
        latency = (end_time - start_time) * 1000 - len(audio)
        print(f"  Playback latency: {latency:.1f}ms")
        print("✓ Buffered audio playback successful")
        return True

    except Exception as e:
        print(f"✗ Buffered audio playback failed: {e}")
        return False

def test_streaming_simulation():
    """Test streaming audio playback simulation with buffering."""
    print("\nTesting streaming audio simulation...")

    try:
        import sounddevice as sd
        import numpy as np
        from pydub import AudioSegment
        from pydub.generators import Sine
        import io
        import time
        import queue
        import threading

        # Generate multiple short audio chunks to simulate streaming
        chunks = []
        for freq in [440, 523, 659, 784]:  # A, C, E, G notes
            tone = Sine(freq).to_audio_segment(duration=500)  # 500ms each
            chunk_bytes = io.BytesIO()
            tone.export(chunk_bytes, format="mp3")
            chunks.append(chunk_bytes.getvalue())

        print(f"  Generated {len(chunks)} audio chunks for streaming test")

        # Audio streaming setup
        buffer_size = 1024
        sample_rate = 22050
        channels = 1
        audio_queue = queue.Queue(maxsize=10)
        playback_finished = threading.Event()

        def audio_callback(outdata, frames, time, status):
            """Callback function for streaming audio."""
            if status:
                print(f"    Audio callback status: {status}")

            try:
                data = audio_queue.get_nowait()
                if data is None:  # End of stream signal
                    playback_finished.set()
                    raise sd.CallbackStop

                # Ensure data fits the buffer
                if len(data) < frames:
                    padding = np.zeros((frames - len(data), channels), dtype=np.float32)
                    data = np.concatenate([data, padding.flatten()])
                elif len(data) > frames:
                    data = data[:frames]

                outdata[:] = data.reshape(frames, channels)

            except queue.Empty:
                outdata.fill(0)

        def stream_processor():
            """Process audio chunks and queue them."""
            try:
                for i, chunk_data in enumerate(chunks):
                    print(f"    Processing chunk {i+1}/{len(chunks)}...")

                    # Convert chunk to AudioSegment
                    audio_segment = AudioSegment.from_file(io.BytesIO(chunk_data), format="mp3")

                    # Set sample rate from first chunk
                    nonlocal sample_rate, channels
                    if i == 0:
                        sample_rate = audio_segment.frame_rate
                        channels = audio_segment.channels

                    # Convert to numpy array
                    audio_data = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                    audio_data = audio_data / np.iinfo(audio_segment.array_type).max

                    # Split into buffer-sized chunks
                    for j in range(0, len(audio_data), buffer_size):
                        chunk = audio_data[j:j + buffer_size]
                        audio_queue.put(chunk, timeout=5.0)

                # Signal end of stream
                audio_queue.put(None)

            except Exception as e:
                print(f"    Error in stream processing: {e}")
                audio_queue.put(None)

        # Start stream processor
        processor_thread = threading.Thread(target=stream_processor)
        processor_thread.daemon = True
        processor_thread.start()

        # Wait for first chunk to set audio parameters
        time.sleep(0.1)

        print("  Playing streaming audio...")

        # Create and start audio stream
        with sd.OutputStream(
            samplerate=sample_rate,
            channels=channels,
            callback=audio_callback,
            blocksize=buffer_size,
            dtype=np.float32
        ):
            playback_finished.wait(timeout=30)

        print("✓ Streaming audio simulation successful")
        return True

    except Exception as e:
        print(f"✗ Streaming audio simulation failed: {e}")
        return False

def test_audio_devices():
    """Test available audio devices."""
    print("\nTesting audio devices...")

    try:
        import sounddevice as sd

        devices = sd.query_devices()
        print("Available audio devices:")

        for i, device in enumerate(devices):
            device_type = []
            if device['max_inputs'] > 0:
                device_type.append("input")
            if device['max_outputs'] > 0:
                device_type.append("output")

            print(f"  {i}: {device['name']} ({'/'.join(device_type)})")

        default_device = sd.default.device
        print(f"  Default device: {default_device}")

        return True

    except Exception as e:
        print(f"✗ Audio device query failed: {e}")
        return False

def main():
    """Run all audio playback tests."""
    print("=== HoldTranscribe Audio Playback Test ===\n")

    # Test dependencies
    if not test_dependencies():
        print("\n✗ Dependencies missing. Please install required packages.")
        return 1

    # Test audio devices
    test_audio_devices()

    # Create test audio
    test_file = create_test_audio()
    if not test_file:
        print("\n✗ Cannot create test audio file")
        return 1

    try:
        # Test direct playback
        direct_success = test_direct_audio_playback(test_file)

        # Test buffered playback
        buffered_success = test_buffered_audio_playback(test_file)

        # Test streaming simulation
        streaming_success = test_streaming_simulation()

        # Summary
        print("\n=== Test Results ===")
        print(f"Direct playback: {'✓ PASS' if direct_success else '✗ FAIL'}")
        print(f"Buffered playback: {'✓ PASS' if buffered_success else '✗ FAIL'}")
        print(f"Streaming simulation: {'✓ PASS' if streaming_success else '✗ FAIL'}")

        if direct_success and buffered_success and streaming_success:
            print("\n✓ All audio playback tests passed!")
            return 0
        else:
            print("\n✗ Some audio playback tests failed")
            return 1

    finally:
        # Clean up test file
        if test_file and os.path.exists(test_file):
            try:
                os.remove(test_file)
                print(f"\nCleaned up test file: {test_file}")
            except Exception as e:
                print(f"\nWarning: Could not clean up test file {test_file}: {e}")

if __name__ == "__main__":
    sys.exit(main())
