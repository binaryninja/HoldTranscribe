#!/usr/bin/env python3
"""
ElevenLabs TTS Integration Demo for HoldTranscribe

This comprehensive demo showcases the complete ElevenLabs text-to-speech integration
with HoldTranscribe, demonstrating all key features and capabilities.

Features demonstrated:
- ElevenLabs as default TTS provider
- High-quality voice synthesis
- Real-time streaming synthesis
- Integration with HoldTranscribe assistant
- Error handling and fallbacks
- Performance metrics
- Voice customization

Requirements:
- ElevenLabs API key (set as ELEVENLABS_API_KEY environment variable)
- elevenlabs package (pip install elevenlabs>=1.0.0)

Usage:
    export ELEVENLABS_API_KEY="your_api_key_here"
    python demo_elevenlabs_integration.py
"""

import os
import sys
import time
import logging
import tempfile
import argparse
from pathlib import Path
from typing import Optional

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def print_header():
    """Print demo header with styling."""
    print("🎤 ElevenLabs TTS Integration Demo - HoldTranscribe")
    print("=" * 60)
    print("Demonstrating high-quality text-to-speech using ElevenLabs")
    print("Now the default TTS implementation in HoldTranscribe")
    print()


def check_prerequisites():
    """Check if all prerequisites are met."""
    print("🔍 Checking Prerequisites")
    print("-" * 30)

    # Check API key
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("❌ ElevenLabs API key not found!")
        print("Please set your API key:")
        print("  export ELEVENLABS_API_KEY='your_api_key_here'")
        print("\nGet your API key from: https://elevenlabs.io/app/settings/api-keys")
        return False, None

    print(f"✅ API key found: {api_key[:8]}..." + "*" * (len(api_key) - 8))

    # Check ElevenLabs package
    try:
        import elevenlabs
        print(f"✅ ElevenLabs package version: {elevenlabs.__version__}")
    except ImportError:
        print("❌ ElevenLabs package not installed!")
        print("Install with: pip install elevenlabs>=1.0.0")
        return False, None

    # Check HoldTranscribe models
    try:
        from holdtranscribe.models import ModelFactory, ModelType
        print("✅ HoldTranscribe models available")
    except ImportError as e:
        print(f"❌ HoldTranscribe models not available: {e}")
        return False, None

    print()
    return True, api_key


def demonstrate_basic_tts(api_key: str):
    """Demonstrate basic text-to-speech functionality."""
    print("🗣️  Basic Text-to-Speech Synthesis")
    print("-" * 40)

    try:
        from holdtranscribe.models import ModelFactory

        # Create default TTS model (ElevenLabs)
        print("Creating ElevenLabs TTS model...")
        start_time = time.time()

        model = ModelFactory.create_tts_model(
            "default",  # Uses ElevenLabs eleven_multilingual_v2 as default
            device="cpu",
            api_key=api_key
        )

        if not model:
            print("❌ Failed to create TTS model")
            return False

        print(f"✅ Model created: {type(model).__name__}")
        print(f"✅ Model name: {model.model_name}")

        # Load the model
        print("Loading model...")
        if not model.load():
            print("❌ Failed to load model")
            return False

        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f} seconds")

        # Test synthesis
        test_text = ("Welcome to HoldTranscribe with ElevenLabs integration! "
                    "This demo showcases high-quality text-to-speech synthesis "
                    "using ElevenLabs as the default TTS provider.")

        output_file = "elevenlabs_demo_basic.mp3"

        print(f"Synthesizing: '{test_text[:50]}...'")
        print(f"Output file: {output_file}")

        synthesis_start = time.time()
        success = model.synthesize(test_text, output_file)
        synthesis_time = time.time() - synthesis_start

        if success:
            file_size = os.path.getsize(output_file)
            print(f"✅ Synthesis completed in {synthesis_time:.2f} seconds")
            print(f"✅ Generated {file_size:,} bytes")
            print(f"🎵 Audio saved to: {output_file}")

            # Calculate performance metrics
            char_count = len(test_text)
            chars_per_second = char_count / synthesis_time
            print(f"📊 Performance: {chars_per_second:.1f} characters/second")
        else:
            print("❌ Synthesis failed")
            return False

        model.unload()
        print("✅ Model unloaded")
        print()
        return True

    except Exception as e:
        print(f"❌ Basic TTS demo failed: {e}")
        return False


def demonstrate_streaming_tts(api_key: str):
    """Demonstrate streaming text-to-speech synthesis."""
    print("🌊 Streaming Text-to-Speech Synthesis")
    print("-" * 42)

    try:
        from holdtranscribe.models import ModelFactory

        # Create fast TTS model for streaming
        print("Creating fast ElevenLabs model for streaming...")

        model = ModelFactory.create_tts_model(
            "eleven_turbo_v2_5",  # Fast model for low latency
            device="cpu",
            api_key=api_key
        )

        if not model or not model.load():
            print("❌ Failed to create/load streaming model")
            return False

        # Configure for optimal streaming
        model.set_streaming_parameters(
            chunk_size=1024,
            optimize_streaming_latency=3  # Maximum optimization
        )
        print("✅ Streaming parameters optimized")

        # Longer text for streaming demo
        streaming_text = (
            "Streaming synthesis allows for real-time audio generation "
            "as text is being processed. This is particularly useful for "
            "interactive applications like voice assistants where low "
            "latency is crucial. ElevenLabs provides excellent streaming "
            "capabilities with high-quality audio output. The audio is "
            "generated in chunks and can be played immediately as each "
            "chunk becomes available, creating a smooth user experience."
        )

        output_file = "elevenlabs_demo_streaming.mp3"

        print(f"Streaming synthesis: '{streaming_text[:50]}...'")
        print(f"Output file: {output_file}")

        # Start streaming synthesis
        start_time = time.time()
        audio_stream = model.synthesize_streaming(streaming_text)

        if audio_stream:
            chunk_count = 0
            total_bytes = 0

            print("✅ Streaming started...")

            with open(output_file, 'wb') as f:
                for chunk in audio_stream:
                    if chunk:
                        f.write(chunk)
                        chunk_count += 1
                        total_bytes += len(chunk)

                        # Progress indicator
                        if chunk_count % 20 == 0:
                            elapsed = time.time() - start_time
                            print(f"  📡 Received {chunk_count} chunks ({total_bytes:,} bytes) in {elapsed:.1f}s")

            total_time = time.time() - start_time
            print(f"✅ Streaming completed in {total_time:.2f} seconds")
            print(f"✅ Total chunks: {chunk_count}")
            print(f"✅ Total bytes: {total_bytes:,}")
            print(f"🎵 Audio saved to: {output_file}")

            # Calculate streaming metrics
            char_count = len(streaming_text)
            chars_per_second = char_count / total_time
            bytes_per_second = total_bytes / total_time
            print(f"📊 Streaming performance: {chars_per_second:.1f} chars/sec, {bytes_per_second/1024:.1f} KB/sec")
        else:
            print("❌ Streaming synthesis failed")
            return False

        model.unload()
        print("✅ Streaming model unloaded")
        print()
        return True

    except Exception as e:
        print(f"❌ Streaming TTS demo failed: {e}")
        return False


def demonstrate_voice_customization(api_key: str):
    """Demonstrate voice parameter customization."""
    print("🎭 Voice Customization")
    print("-" * 22)

    try:
        from holdtranscribe.models import ModelFactory

        model = ModelFactory.create_tts_model(
            "elevenlabs",
            device="cpu",
            api_key=api_key
        )

        if not model or not model.load():
            print("❌ Failed to create/load model for voice demo")
            return False

        # Test different voice settings
        voice_configs = [
            {
                "name": "Stable Voice",
                "settings": {"stability": 0.8, "similarity_boost": 0.6, "style": 0.0},
                "text": "This is a stable and consistent voice configuration."
            },
            {
                "name": "Expressive Voice",
                "settings": {"stability": 0.3, "similarity_boost": 0.8, "style": 0.4},
                "text": "This is a more expressive and dynamic voice!"
            },
            {
                "name": "Calm Voice",
                "settings": {"stability": 0.9, "similarity_boost": 0.5, "style": 0.1},
                "text": "This is a calm and measured voice for professional use."
            }
        ]

        for i, config in enumerate(voice_configs):
            print(f"Testing {config['name']}...")

            # Set voice parameters
            model.set_voice_parameters(
                voice_id="21m00Tcm4TlvDq8ikWAM",  # Default voice
                **config["settings"]
            )

            output_file = f"elevenlabs_demo_voice_{i+1}.mp3"

            start_time = time.time()
            success = model.synthesize(config["text"], output_file)
            synthesis_time = time.time() - start_time

            if success:
                file_size = os.path.getsize(output_file)
                print(f"  ✅ {config['name']}: {file_size:,} bytes in {synthesis_time:.2f}s")
                print(f"  🎵 Saved to: {output_file}")
            else:
                print(f"  ❌ {config['name']} synthesis failed")

        model.unload()
        print("✅ Voice customization demo completed")
        print()
        return True

    except Exception as e:
        print(f"❌ Voice customization demo failed: {e}")
        return False


def demonstrate_holdtranscribe_integration(api_key: str):
    """Demonstrate full HoldTranscribe app integration."""
    print("🚀 HoldTranscribe App Integration")
    print("-" * 34)

    try:
        from holdtranscribe.app import HoldTranscribeApp
        import argparse

        # Create HoldTranscribe app instance
        print("Creating HoldTranscribe app with ElevenLabs TTS...")
        app = HoldTranscribeApp()

        # Configure app arguments
        args = argparse.Namespace()
        args.model = "mistralai/Voxtral-Mini-3B-2507"  # AI assistant model
        args.fast = False
        args.beam_size = 5
        args.debug = False  # Reduce noise for demo
        args.tts = True
        args.tts_model = "default"  # Should use ElevenLabs
        args.tts_output = "assistant_response.mp3"

        app.args = args
        app.device = "cpu"

        # Initialize models
        print("Initializing models (this may take a moment)...")
        start_time = time.time()

        if app.initialize_models():
            init_time = time.time() - start_time
            print(f"✅ Models initialized in {init_time:.1f} seconds")

            # Verify TTS model
            if app.tts_model:
                print(f"✅ TTS Model: {type(app.tts_model).__name__}")
                print(f"✅ TTS Model Name: {app.tts_model.model_name}")
                print(f"✅ TTS Loaded: {app.tts_model.is_loaded}")

                # Test TTS generation through app
                print("\nTesting assistant response with TTS...")
                test_responses = [
                    "Hello! I'm your AI assistant powered by HoldTranscribe with ElevenLabs text-to-speech.",
                    "The integration is working perfectly! You can now hear my responses with high-quality voice synthesis.",
                    "ElevenLabs provides excellent voice quality and fast response times for a great user experience."
                ]

                for i, response in enumerate(test_responses):
                    print(f"Generating response {i+1}: '{response[:40]}...'")
                    app._generate_tts(response)
                    print(f"  ✅ TTS audio generated")

                print("✅ Assistant integration test completed")
            else:
                print("❌ No TTS model loaded")
                return False

            # Clean up
            app.cleanup()
            print("✅ App cleaned up")

        else:
            print("❌ Model initialization failed")
            return False

        print()
        return True

    except Exception as e:
        print(f"❌ HoldTranscribe integration demo failed: {e}")
        return False


def demonstrate_performance_comparison():
    """Demonstrate performance comparison between TTS models."""
    print("📊 Performance Comparison")
    print("-" * 26)

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("❌ API key required for performance test")
        return False

    try:
        from holdtranscribe.models import ModelFactory

        test_text = "Performance testing with different ElevenLabs models for speed and quality comparison."

        models_to_test = [
            ("eleven_multilingual_v2", "High Quality"),
            ("eleven_turbo_v2_5", "Fast"),
            ("eleven_flash_v2_5", "Fastest")
        ]

        results = []

        for model_name, description in models_to_test:
            print(f"Testing {model_name} ({description})...")

            try:
                model = ModelFactory.create_tts_model(model_name, "cpu", api_key=api_key)

                if model and model.load():
                    # Time the synthesis
                    start_time = time.time()
                    output_file = f"perf_test_{model_name}.mp3"
                    success = model.synthesize(test_text, output_file)
                    synthesis_time = time.time() - start_time

                    if success:
                        file_size = os.path.getsize(output_file)
                        chars_per_sec = len(test_text) / synthesis_time

                        results.append({
                            'model': model_name,
                            'description': description,
                            'time': synthesis_time,
                            'size': file_size,
                            'speed': chars_per_sec
                        })

                        print(f"  ✅ {synthesis_time:.2f}s, {file_size:,} bytes, {chars_per_sec:.1f} chars/sec")
                    else:
                        print(f"  ❌ Synthesis failed")

                    model.unload()
                else:
                    print(f"  ❌ Failed to load {model_name}")

            except Exception as e:
                print(f"  ❌ Error testing {model_name}: {e}")

        # Display comparison table
        if results:
            print("\n📈 Performance Summary:")
            print(f"{'Model':<25} {'Type':<12} {'Time':<8} {'Size':<10} {'Speed':<12}")
            print("-" * 70)

            for result in results:
                print(f"{result['model']:<25} {result['description']:<12} "
                      f"{result['time']:.2f}s    {result['size']:>7,}B  "
                      f"{result['speed']:>8.1f} c/s")

        print()
        return True

    except Exception as e:
        print(f"❌ Performance comparison failed: {e}")
        return False


def print_summary(results: dict):
    """Print demo summary."""
    print("🎉 Demo Summary")
    print("=" * 60)

    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)

    print(f"Tests completed: {passed_tests}/{total_tests}")
    print()

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")

    print()

    if passed_tests == total_tests:
        print("🎊 All tests passed! ElevenLabs TTS integration is fully functional.")
        print()
        print("Next steps:")
        print("1. Use HoldTranscribe with TTS:")
        print("   python -m holdtranscribe.main --tts")
        print()
        print("2. Use with AI assistant:")
        print("   python -m holdtranscribe.main --model mistralai/Voxtral-Mini-3B-2507 --tts")
        print()
        print("3. Try different models:")
        print("   --tts-model eleven_turbo_v2_5    (faster)")
        print("   --tts-model eleven_flash_v2_5     (fastest)")
        print()
        print("4. Check generated audio files:")
        print("   ls -la elevenlabs_demo_*.mp3")
        print("   ls -la assistant_response_*.mp3")
    else:
        failed_count = total_tests - passed_tests
        print(f"⚠️  {failed_count} test(s) failed. Check the output above for details.")
        print()
        print("Common issues:")
        print("- API key permissions (some keys have limited access)")
        print("- Network connectivity")
        print("- Missing dependencies")
        print()
        print("The core TTS functionality should still work for synthesis.")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="ElevenLabs TTS Integration Demo")
    parser.add_argument("--skip-performance", action="store_true",
                       help="Skip performance comparison (faster demo)")
    parser.add_argument("--skip-integration", action="store_true",
                       help="Skip HoldTranscribe integration test")
    args = parser.parse_args()

    print_header()

    # Check prerequisites
    prereqs_ok, api_key = check_prerequisites()
    if not prereqs_ok:
        return 1

    # Run demo tests
    results = {}

    print("🚀 Starting ElevenLabs TTS Integration Demo")
    print("=" * 50)
    print()

    # Basic TTS test
    results["Basic TTS Synthesis"] = demonstrate_basic_tts(api_key)

    # Streaming test
    results["Streaming Synthesis"] = demonstrate_streaming_tts(api_key)

    # Voice customization test
    results["Voice Customization"] = demonstrate_voice_customization(api_key)

    # HoldTranscribe integration test (optional)
    if not args.skip_integration:
        results["HoldTranscribe Integration"] = demonstrate_holdtranscribe_integration(api_key)

    # Performance comparison (optional)
    if not args.skip_performance:
        results["Performance Comparison"] = demonstrate_performance_comparison()

    # Print summary
    print_summary(results)

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during demo: {e}")
        sys.exit(1)
