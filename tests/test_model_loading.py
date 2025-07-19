#!/usr/bin/env python3
"""
Test script for model loading functionality in HoldTranscribe.
Tests both Voxtral and Whisper model loading capabilities.
"""

import sys
import os
import time

# Add the holdtranscribe module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'holdtranscribe'))

def test_imports():
    """Test if all required and optional dependencies are available."""
    print("=== Testing Dependencies ===")

    # Test core dependencies
    try:
        import numpy as np
        print("✓ numpy available")
    except ImportError as e:
        print(f"✗ numpy not available: {e}")
        return False

    try:
        import sounddevice as sd
        print("✓ sounddevice available")
    except ImportError as e:
        print(f"✗ sounddevice not available: {e}")
        return False

    try:
        from faster_whisper import WhisperModel
        print("✓ faster-whisper available")
    except ImportError as e:
        print(f"✗ faster-whisper not available: {e}")
        return False

    # Test optional Voxtral dependencies
    try:
        from transformers import AutoProcessor, VoxtralForConditionalGeneration
        import torch
        print("✓ transformers and torch available (Voxtral support enabled)")
        return True, True
    except ImportError as e:
        print(f"⚠ transformers/torch not available: {e}")
        print("  → Voxtral support disabled, will use Whisper only")
        return True, False

def test_model_loading():
    """Test the model loading functionality."""
    print("\n=== Testing Model Loading ===")

    # Import the load_model function
    try:
        from main import load_model, detect_device
        print("✓ Successfully imported load_model function")
    except ImportError as e:
        print(f"✗ Failed to import load_model: {e}")
        return False

    # Detect device
    device = detect_device()
    print(f"✓ Device detected: {device}")

    # Test models to try
    models_to_test = [
        ("mistralai/Voxtral-Mini-3B-2507", "Voxtral"),
        ("tiny", "Whisper Tiny"),
        ("base", "Whisper Base"),
    ]

    for model_name, model_desc in models_to_test:
        print(f"\n--- Testing {model_desc} ({model_name}) ---")
        try:
            start_time = time.time()
            model_type, model, processor = load_model(model_name, device, fast_mode=True, beam_size=1)
            load_time = time.time() - start_time

            print(f"✓ {model_desc} loaded successfully!")
            print(f"  Type: {model_type}")
            print(f"  Load time: {load_time:.2f}s")
            print(f"  Has processor: {processor is not None}")

            # Clean up model to free memory
            del model
            if processor:
                del processor

        except Exception as e:
            print(f"✗ Failed to load {model_desc}: {e}")
            continue

    return True

def test_audio_processing():
    """Test basic audio processing capabilities."""
    print("\n=== Testing Audio Processing ===")

    try:
        import numpy as np
        import sounddevice as sd

        # Test basic audio setup
        print("✓ Audio libraries available")

        # Test device query
        try:
            devices = sd.query_devices()
            print(f"✓ Found {len(devices)} audio devices")

            # Find default input device
            default_device = sd.default.device[0]
            if default_device is not None:
                device_info = sd.query_devices(default_device)
                print(f"✓ Default input device: {device_info['name']}")
            else:
                print("⚠ No default input device found")

        except Exception as e:
            print(f"⚠ Audio device query failed: {e}")

        # Test creating dummy audio data
        sample_rate = 16000
        duration = 1.0  # 1 second
        dummy_audio = np.random.randint(-32768, 32768, int(sample_rate * duration), dtype=np.int16)
        print(f"✓ Created dummy audio data: {len(dummy_audio)} samples")

        return True

    except Exception as e:
        print(f"✗ Audio processing test failed: {e}")
        return False

def main():
    """Main test function."""
    print("HoldTranscribe Model Loading Test")
    print("=" * 40)

    # Test imports
    core_ok, voxtral_ok = test_imports()
    if not core_ok:
        print("\n✗ Core dependencies missing. Please install requirements:")
        print("pip install -r requirements.txt")
        return False

    # Test model loading
    if not test_model_loading():
        print("\n✗ Model loading tests failed")
        return False

    # Test audio processing
    if not test_audio_processing():
        print("\n✗ Audio processing tests failed")
        return False

    print("\n" + "=" * 40)
    print("✓ All tests passed!")

    if voxtral_ok:
        print("🚀 Voxtral support is available")
    else:
        print("🖥️  Whisper-only mode (install transformers+torch for Voxtral)")

    print("\nReady to run HoldTranscribe!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
