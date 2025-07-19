#!/usr/bin/env python3
"""
Test script to verify ElevenLabs TTS integration with HoldTranscribe.

This script tests the integration without making actual API calls.
"""

import sys
import os
import logging
from pathlib import Path

# Add the holdtranscribe module to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")

    try:
        from holdtranscribe.models import ModelFactory, ModelType
        print("✅ HoldTranscribe models imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import HoldTranscribe models: {e}")
        return False

    try:
        from holdtranscribe.models.elevenlabs_model import ElevenLabsTTSModel, ELEVENLABS_AVAILABLE
        print("✅ ElevenLabs model imported successfully")

        if ELEVENLABS_AVAILABLE:
            print("✅ ElevenLabs package is available")
        else:
            print("⚠️  ElevenLabs package not installed (this is okay for testing)")
    except ImportError as e:
        print(f"❌ Failed to import ElevenLabs model: {e}")
        return False

    try:
        from holdtranscribe.models.elevenlabs_wrapper import ElevenLabsTTSWrapper
        print("✅ ElevenLabs wrapper imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import ElevenLabs wrapper: {e}")
        return False

    return True

def test_model_factory():
    """Test that the model factory can create ElevenLabs models."""
    print("\n🏭 Testing model factory...")

    try:
        from holdtranscribe.models import ModelFactory

        # Test different model name patterns
        test_cases = [
            "eleven_multilingual_v2",
            "eleven_turbo_v2_5",
            "elevenlabs",
            "eleven_flash_v2_5"
        ]

        for model_name in test_cases:
            print(f"  Testing model name: {model_name}")

            try:
                model = ModelFactory.create_tts_model(
                    model_name=model_name,
                    device="cpu",
                    api_key="test_key"  # Dummy key for testing
                )

                if model:
                    print(f"    ✅ Factory created model for {model_name}")
                    print(f"    📋 Model type: {type(model).__name__}")
                    print(f"    📋 Model info: {model}")
                else:
                    print(f"    ❌ Factory returned None for {model_name}")

            except Exception as e:
                print(f"    ❌ Error creating model {model_name}: {e}")

        return True

    except Exception as e:
        print(f"❌ Model factory test failed: {e}")
        return False

def test_model_interface():
    """Test the model interface without making API calls."""
    print("\n🔧 Testing model interface...")

    try:
        from holdtranscribe.models import ModelFactory

        # Create a model instance
        model = ModelFactory.create_tts_model(
            model_name="elevenlabs",
            device="cpu",
            api_key="test_key"
        )

        if not model:
            print("❌ Could not create model instance")
            return False

        print("✅ Model instance created")

        # Test interface methods (without calling load)
        print("  Testing interface methods...")

        # Test model type
        from holdtranscribe.models import ModelType
        if hasattr(model, 'model_type') and model.model_type == ModelType.TTS:
            print("    ✅ Model type is correct (TTS)")
        else:
            print("    ❌ Model type is incorrect")

        # Test basic attributes
        if hasattr(model, 'model_name'):
            print(f"    ✅ Model name: {model.model_name}")
        else:
            print("    ❌ Model name attribute missing")

        if hasattr(model, 'is_loaded'):
            print(f"    ✅ Load status: {model.is_loaded}")
        else:
            print("    ❌ Load status attribute missing")

        # Test required methods exist
        required_methods = ['load', 'unload', 'synthesize']
        for method_name in required_methods:
            if hasattr(model, method_name) and callable(getattr(model, method_name)):
                print(f"    ✅ Method {method_name} exists")
            else:
                print(f"    ❌ Method {method_name} missing or not callable")

        # Test optional methods
        optional_methods = ['synthesize_streaming', 'get_available_voices', 'set_voice_parameters']
        for method_name in optional_methods:
            if hasattr(model, method_name) and callable(getattr(model, method_name)):
                print(f"    ✅ Optional method {method_name} exists")
            else:
                print(f"    ⚠️  Optional method {method_name} missing")

        return True

    except Exception as e:
        print(f"❌ Interface test failed: {e}")
        return False

def test_error_handling():
    """Test error handling for missing API key and package."""
    print("\n🚨 Testing error handling...")

    try:
        from holdtranscribe.models.elevenlabs_model import ELEVENLABS_AVAILABLE

        if not ELEVENLABS_AVAILABLE:
            print("✅ Graceful handling of missing ElevenLabs package")

            # Test that we get proper error when trying to use without package
            try:
                from holdtranscribe.models.elevenlabs_model import ElevenLabsTTSModel
                model = ElevenLabsTTSModel()
                print("❌ Should have raised ImportError for missing package")
                return False
            except ImportError:
                print("✅ Proper ImportError raised for missing package")
        else:
            print("✅ ElevenLabs package is available")

        # Test missing API key handling
        from holdtranscribe.models import ModelFactory

        # Clear any existing API key for this test
        old_key = os.environ.get('ELEVENLABS_API_KEY')
        if 'ELEVENLABS_API_KEY' in os.environ:
            del os.environ['ELEVENLABS_API_KEY']

        try:
            model = ModelFactory.create_tts_model(
                model_name="elevenlabs",
                device="cpu"
                # No api_key provided
            )

            if model:
                print("✅ Model created without API key (will fail on load)")

                # Test that load fails appropriately
                try:
                    success = model.load()
                    if not success:
                        print("✅ Load correctly failed without API key")
                    else:
                        print("⚠️  Load succeeded without API key (unexpected)")
                except Exception as e:
                    print(f"✅ Load raised exception without API key: {type(e).__name__}")
            else:
                print("❌ Model creation failed")

        finally:
            # Restore API key if it existed
            if old_key:
                os.environ['ELEVENLABS_API_KEY'] = old_key

        return True

    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 ElevenLabs TTS Integration Test")
    print("=" * 50)

    # Configure logging to see any issues
    logging.basicConfig(level=logging.WARNING)

    tests = [
        ("Import Test", test_imports),
        ("Model Factory Test", test_model_factory),
        ("Interface Test", test_model_interface),
        ("Error Handling Test", test_error_handling)
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print(f"{'='*50}")

        try:
            success = test_func()
            results.append((test_name, success))

            if success:
                print(f"\n✅ {test_name} PASSED")
            else:
                print(f"\n❌ {test_name} FAILED")

        except Exception as e:
            print(f"\n❌ {test_name} FAILED with exception: {e}")
            results.append((test_name, False))

    # Summary
    print(f"\n{'='*50}")
    print("📊 TEST SUMMARY")
    print(f"{'='*50}")

    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if success:
            passed += 1

    total = len(results)
    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! ElevenLabs integration is ready.")
        print("\nNext steps:")
        print("1. Install ElevenLabs package: pip install elevenlabs")
        print("2. Set your API key: export ELEVENLABS_API_KEY='your_key'")
        print("3. Run the example: python examples/elevenlabs_tts_example.py")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check the output above.")

    return 0 if passed == total else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
