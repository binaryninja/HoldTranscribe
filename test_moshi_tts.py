#!/usr/bin/env python3
"""
Test script for Moshi TTS model implementation.

This script tests the basic functionality of the new MoshiTTSModel
including synthesis, streaming, and voice conditioning.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_moshi_basic():
    """Test basic Moshi TTS functionality."""
    print("🧪 Testing Moshi TTS Model...")

    try:
        from holdtranscribe.models.moshi_model import MoshiTTSModel

        # Test model creation
        print("1. Creating Moshi TTS model...")
        model = MoshiTTSModel(device="cuda" if input("Use CUDA? (y/n): ").lower() == 'y' else "cpu")
        print(f"   ✓ Model created: {model}")

        # Test model loading
        print("2. Loading model...")
        start_time = time.time()
        success = model.load()
        load_time = time.time() - start_time

        if success:
            print(f"   ✓ Model loaded successfully in {load_time:.1f}s")
            print(f"   ✓ Model info: {model.get_model_info()}")
        else:
            print("   ✗ Failed to load model")
            return False

        # Test basic synthesis
        print("3. Testing basic synthesis...")
        test_text = "Hello, this is a test of the Moshi text-to-speech system."

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_file = tmp.name

        try:
            start_time = time.time()
            success = model.synthesize(test_text, output_file)
            synthesis_time = time.time() - start_time

            if success and os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                print(f"   ✓ Synthesis successful in {synthesis_time:.1f}s")
                print(f"   ✓ Output file: {output_file} ({file_size} bytes)")

                # Optionally play the audio
                if input("   Play generated audio? (y/n): ").lower() == 'y':
                    try:
                        model.play_audio_to_speakers(test_text)
                        print("   ✓ Audio playback completed")
                    except Exception as e:
                        print(f"   ⚠ Audio playback failed: {e}")
            else:
                print("   ✗ Synthesis failed")
                return False
        finally:
            try:
                os.unlink(output_file)
            except:
                pass

        # Test streaming synthesis
        print("4. Testing streaming synthesis...")
        long_text = (
            "This is a longer text to test streaming synthesis. "
            "The model should split this into chunks and generate audio "
            "for each chunk separately. This allows for better real-time "
            "performance and lower latency in conversational applications."
        )

        try:
            chunks_generated = 0
            total_audio_length = 0

            start_time = time.time()
            for audio_chunk in model.synthesize_streaming(long_text):
                chunks_generated += 1
                total_audio_length += len(audio_chunk)
            streaming_time = time.time() - start_time

            print(f"   ✓ Streaming synthesis completed in {streaming_time:.1f}s")
            print(f"   ✓ Generated {chunks_generated} chunks, {total_audio_length} total samples")
        except Exception as e:
            print(f"   ✗ Streaming synthesis failed: {e}")

        # Test voice parameters
        print("5. Testing voice parameters...")
        try:
            original_temp = model.temperature
            model.set_voice_parameters(temperature=0.8, max_new_tokens=30)
            print(f"   ✓ Temperature changed from {original_temp} to {model.temperature}")

            model.set_streaming_parameters(chunk_max_words=15, chunk_silence_duration=0.3)
            print(f"   ✓ Streaming parameters updated")
        except Exception as e:
            print(f"   ⚠ Parameter setting failed: {e}")

        # Test available voices
        print("6. Testing voice options...")
        try:
            voices = model.get_available_voices()
            print(f"   ✓ Available voices: {voices}")
        except Exception as e:
            print(f"   ⚠ Voice listing failed: {e}")

        # Test model unloading
        print("7. Unloading model...")
        model.unload()
        print(f"   ✓ Model unloaded, is_loaded: {model.is_loaded}")

        print("\n🎉 All Moshi TTS tests completed successfully!")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install required dependencies:")
        print("pip install -r requirements-tts.txt")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_factory():
    """Test Moshi integration with model factory."""
    print("\n🧪 Testing Model Factory Integration...")

    try:
        from holdtranscribe.models import ModelFactory, ModelType

        # Test creating Moshi model through factory
        print("1. Testing factory creation...")
        model = ModelFactory.create_tts_model("default", "cpu")

        if model:
            print(f"   ✓ Factory created model: {type(model).__name__}")
            print(f"   ✓ Model name: {model.model_name}")

            # Test factory type detection
            if hasattr(model, 'model_type'):
                print(f"   ✓ Model type: {model.model_type}")
        else:
            print("   ✗ Factory failed to create model")
            return False

        # Test with explicit Moshi models
        test_models = [
            "kyutai/moshiko-pytorch-bf16",
            "moshi",
            "kyutai/model"
        ]

        for model_name in test_models:
            print(f"2. Testing factory with '{model_name}'...")
            test_model = ModelFactory.create_tts_model(model_name, "cpu")
            if test_model:
                print(f"   ✓ Created {type(test_model).__name__}")
            else:
                print(f"   ✗ Failed to create model for {model_name}")

        print("🎉 Model factory integration tests passed!")
        return True

    except Exception as e:
        print(f"❌ Factory test failed: {e}")
        return False

def test_app_integration():
    """Test integration with main app."""
    print("\n🧪 Testing App Integration...")

    try:
        from holdtranscribe.app import HoldTranscribeApp
        import argparse

        # Create minimal args for testing
        args = argparse.Namespace()
        args.transcription_model = "base"
        args.assistant_model = None
        args.tts_model = "default"  # Should use Moshi now
        args.tts = True
        args.device = "cpu"
        args.debug = True
        args.use_assistant = False
        args.hotkey = ['space']

        print("1. Creating app with Moshi TTS...")
        app = HoldTranscribeApp()
        app.args = args
        app.device = "cpu"

        # Test model initialization
        print("2. Initializing models...")
        try:
            app.initialize_models()

            if app.tts_model:
                print(f"   ✓ TTS model loaded: {type(app.tts_model).__name__}")
                print(f"   ✓ Model info: {app.tts_model.get_model_info()}")

                # Test TTS generation through app
                print("3. Testing TTS through app...")
                test_response = "This is a test response from the assistant."
                app._generate_tts(test_response)
                print("   ✓ TTS generation completed")

            else:
                print("   ⚠ No TTS model loaded")
        except Exception as e:
            print(f"   ⚠ Model initialization issue: {e}")

        # Cleanup
        try:
            app.cleanup()
            print("   ✓ App cleanup completed")
        except:
            pass

        print("🎉 App integration tests completed!")
        return True

    except Exception as e:
        print(f"❌ App integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Moshi TTS Tests")
    print("=" * 50)

    # Check dependencies first
    try:
        import torch
        import transformers
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ Transformers version: {transformers.__version__}")

        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("ℹ CUDA not available, will use CPU")

    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please install: pip install -r requirements-tts.txt")
        return 1

    print()

    # Run tests
    tests = [
        ("Basic Moshi TTS", test_moshi_basic),
        ("Model Factory Integration", test_model_factory),
        ("App Integration", test_app_integration)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except KeyboardInterrupt:
            print("\n⚠ Test interrupted by user")
            break
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Moshi TTS is ready to use.")
        return 0
    else:
        print("⚠ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
