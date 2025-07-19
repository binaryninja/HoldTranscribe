#!/usr/bin/env python3
"""
Test script for Kyutai TTS implementation in HoldTranscribe.

This script tests the text-to-speech functionality with the Kyutai/Moshi models.
"""

import sys
import os
import time
import tempfile
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")

    try:
        from holdtranscribe.tts import create_tts_model, list_available_models
        print("✅ TTS module imports successful")
        return True
    except ImportError as e:
        print(f"❌ Failed to import TTS modules: {e}")
        return False

def test_model_creation():
    """Test TTS model creation."""
    print("\n🔍 Testing model creation...")

    try:
        from holdtranscribe.tts import create_tts_model

        # Test creating a Kyutai model
        model = create_tts_model(
            model_type="kyutai",
            model_name="default",
            device="cpu"  # Use CPU for testing to avoid CUDA issues
        )

        if model is None:
            print("❌ Failed to create TTS model")
            return False

        print("✅ TTS model created successfully")
        print(f"   Model info: {model}")
        return model

    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_model_loading(model):
    """Test TTS model loading."""
    print("\n🔍 Testing model loading...")

    try:
        print("   Loading model (this may take a while for first time download)...")
        success = model.load()

        if success:
            print("✅ Model loaded successfully")
            print(f"   Model info: {model.get_model_info()}")
            return True
        else:
            print("❌ Model loading failed")
            return False

    except Exception as e:
        print(f"❌ Model loading failed with exception: {e}")
        return False

def test_synthesis_to_file(model):
    """Test text-to-speech synthesis to file."""
    print("\n🔍 Testing synthesis to file...")

    try:
        # Create a temporary output file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            output_path = tmp_file.name

        test_text = "Hello, this is a test of the Kyutai text-to-speech system."
        print(f"   Synthesizing: '{test_text}'")
        print(f"   Output file: {output_path}")

        start_time = time.time()
        success = model.synthesize(test_text, output_file=output_path)
        end_time = time.time()

        if success and os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✅ Synthesis successful!")
            print(f"   Generated {file_size} bytes in {end_time - start_time:.2f} seconds")
            print(f"   Audio file saved: {output_path}")

            # Clean up
            try:
                os.unlink(output_path)
                print("   Cleaned up temporary file")
            except:
                pass

            return True
        else:
            print("❌ Synthesis failed - no output file generated")
            return False

    except Exception as e:
        print(f"❌ Synthesis failed with exception: {e}")
        return False

def test_streaming_synthesis(model):
    """Test streaming synthesis."""
    print("\n🔍 Testing streaming synthesis...")

    try:
        test_text = "This is a longer text to test streaming synthesis. It should be split into multiple chunks for processing."
        print(f"   Streaming synthesis of: '{test_text[:50]}...'")

        chunk_count = 0
        total_audio_length = 0

        start_time = time.time()
        for audio_chunk in model.synthesize_streaming(test_text):
            chunk_count += 1
            if hasattr(audio_chunk, 'shape'):
                total_audio_length += audio_chunk.shape[-1]
            print(f"   Received chunk {chunk_count} with {len(audio_chunk) if hasattr(audio_chunk, '__len__') else 'unknown'} samples")

        end_time = time.time()

        if chunk_count > 0:
            print(f"✅ Streaming synthesis successful!")
            print(f"   Generated {chunk_count} chunks with {total_audio_length} total samples")
            print(f"   Completed in {end_time - start_time:.2f} seconds")
            return True
        else:
            print("❌ No audio chunks generated")
            return False

    except Exception as e:
        print(f"❌ Streaming synthesis failed: {e}")
        return False

def test_speaker_output(model):
    """Test audio output to speakers (optional, requires user confirmation)."""
    print("\n🔍 Testing speaker output...")

    try:
        # Ask user if they want to test speaker output
        response = input("   Do you want to test audio output to speakers? (y/N): ").strip().lower()

        if response not in ['y', 'yes']:
            print("   Skipping speaker output test")
            return True

        test_text = "Hello! If you can hear this, the speaker output is working correctly."
        print(f"   Playing to speakers: '{test_text}'")
        print("   (You should hear audio from your speakers now)")

        start_time = time.time()
        model.play_audio_to_speakers(test_text)
        end_time = time.time()

        print(f"✅ Speaker output completed in {end_time - start_time:.2f} seconds")
        return True

    except ImportError as e:
        print(f"⚠️  Speaker output not available: {e}")
        return True  # Not a failure, just missing optional dependency
    except Exception as e:
        print(f"❌ Speaker output failed: {e}")
        return False

def test_model_parameters(model):
    """Test setting various model parameters."""
    print("\n🔍 Testing model parameters...")

    try:
        # Test setting voice parameters
        model.set_voice_parameters(temp=0.8, cfg_coef=1.5)

        # Test setting streaming parameters
        model.set_streaming_parameters(chunk_max_words=25, chunk_silence_duration=0.3)

        # Test setting seed
        model.set_seed(42)

        print("✅ Model parameters set successfully")
        return True

    except Exception as e:
        print(f"❌ Setting model parameters failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Kyutai TTS tests for HoldTranscribe")
    print("=" * 50)

    # Track test results
    tests_passed = 0
    total_tests = 0

    # Test imports
    total_tests += 1
    if test_imports():
        tests_passed += 1
    else:
        print("\n❌ Import test failed - cannot continue")
        return 1

    # Test model creation
    total_tests += 1
    model = test_model_creation()
    if model:
        tests_passed += 1
    else:
        print("\n❌ Model creation failed - cannot continue")
        return 1

    try:
        # Test model loading
        total_tests += 1
        if test_model_loading(model):
            tests_passed += 1
        else:
            print("\n❌ Model loading failed - skipping remaining tests")
            return 1

        # Test model parameters
        total_tests += 1
        if test_model_parameters(model):
            tests_passed += 1

        # Test synthesis to file
        total_tests += 1
        if test_synthesis_to_file(model):
            tests_passed += 1

        # Test streaming synthesis
        total_tests += 1
        if test_streaming_synthesis(model):
            tests_passed += 1

        # Test speaker output (optional)
        total_tests += 1
        if test_speaker_output(model):
            tests_passed += 1

    finally:
        # Clean up
        try:
            print("\n🧹 Cleaning up...")
            model.unload()
            print("   Model unloaded")
        except:
            pass

    # Print results
    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print("🎉 All tests passed! Kyutai TTS is working correctly.")
        return 0
    else:
        print(f"⚠️  {total_tests - tests_passed} test(s) failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
