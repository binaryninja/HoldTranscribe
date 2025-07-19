#!/usr/bin/env python3
"""
Example script demonstrating ElevenLabs TTS integration with HoldTranscribe.

This example shows how to:
1. Initialize ElevenLabs TTS model
2. Perform basic text-to-speech synthesis
3. Use different voices and settings
4. Stream audio synthesis
5. Handle API key configuration

Requirements:
- ElevenLabs API key (set as environment variable ELEVENLABS_API_KEY)
- elevenlabs package installed (pip install elevenlabs)
"""

import os
import sys
import logging
import tempfile
from pathlib import Path

# Add the parent directory to Python path to import holdtranscribe
sys.path.insert(0, str(Path(__file__).parent.parent))

from holdtranscribe.models import ModelFactory, ModelType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_api_key():
    """Check if ElevenLabs API key is available."""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("❌ ElevenLabs API key not found!")
        print("Please set your API key as an environment variable:")
        print("export ELEVENLABS_API_KEY='your_api_key_here'")
        print("\nYou can get your API key from: https://elevenlabs.io/app/settings/api-keys")
        return False

    print(f"✅ API key found: {api_key[:8]}..." + "*" * (len(api_key) - 8))
    return True


def basic_synthesis_example():
    """Demonstrate basic text-to-speech synthesis."""
    print("\n🗣️  Basic Text-to-Speech Synthesis Example")
    print("=" * 50)

    # Create ElevenLabs TTS model using the factory
    model = ModelFactory.create_tts_model(
        model_name="eleven_multilingual_v2",  # ElevenLabs model
        device="cpu",  # Device parameter (not used for cloud API)
        api_key=os.getenv("ELEVENLABS_API_KEY")
    )

    if not model:
        print("❌ Failed to create ElevenLabs TTS model")
        return False

    # Load the model
    print("Loading ElevenLabs model...")
    if not model.load():
        print("❌ Failed to load ElevenLabs model")
        return False

    print("✅ Model loaded successfully")

    # Text to synthesize
    text = "Hello! This is a test of ElevenLabs text-to-speech integration with HoldTranscribe."

    # Create temporary output file
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        output_path = temp_file.name

    print(f"Synthesizing text: '{text}'")
    print(f"Output will be saved to: {output_path}")

    # Synthesize text to speech
    success = model.synthesize(text, output_path)

    if success:
        print("✅ Synthesis completed successfully!")
        print(f"Audio file saved to: {output_path}")

        # Check file size
        file_size = os.path.getsize(output_path)
        print(f"File size: {file_size} bytes")

        if file_size > 0:
            print("🎵 You can play the audio file with any media player")
        else:
            print("⚠️  Warning: Generated file is empty")
    else:
        print("❌ Synthesis failed!")

    # Unload model
    model.unload()
    return success


def voice_selection_example():
    """Demonstrate using different voices."""
    print("\n🎭 Voice Selection Example")
    print("=" * 50)

    # Create model
    model = ModelFactory.create_tts_model(
        model_name="elevenlabs",
        device="cpu",
        api_key=os.getenv("ELEVENLABS_API_KEY")
    )

    if not model or not model.load():
        print("❌ Failed to create/load model")
        return False

    # Get available voices
    print("Fetching available voices...")
    voices = model.get_available_voices()

    if not voices:
        print("❌ Failed to fetch voices")
        model.unload()
        return False

    print(f"✅ Found {len(voices)} available voices:")

    # Display first 5 voices
    for i, voice in enumerate(voices[:5]):
        print(f"  {i+1}. {voice['name']} (ID: {voice['voice_id']})")
        if voice.get('description'):
            print(f"     Description: {voice['description']}")

    if len(voices) > 5:
        print(f"  ... and {len(voices) - 5} more voices")

    # Use a specific voice for synthesis
    if voices:
        selected_voice = voices[0]  # Use first available voice
        print(f"\nUsing voice: {selected_voice['name']}")

        # Set voice parameters
        model.set_voice_parameters(
            voice_id=selected_voice['voice_id'],
            stability=0.7,
            similarity_boost=0.8
        )

        text = f"Hello! I'm speaking with the {selected_voice['name']} voice from ElevenLabs."

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            output_path = temp_file.name

        success = model.synthesize(text, output_path)

        if success:
            print(f"✅ Voice synthesis completed: {output_path}")
        else:
            print("❌ Voice synthesis failed")

    model.unload()
    return True


def streaming_synthesis_example():
    """Demonstrate streaming synthesis."""
    print("\n🌊 Streaming Synthesis Example")
    print("=" * 50)

    # Create model
    model = ModelFactory.create_tts_model(
        model_name="eleven_turbo_v2_5",  # Fast model for streaming
        device="cpu",
        api_key=os.getenv("ELEVENLABS_API_KEY")
    )

    if not model or not model.load():
        print("❌ Failed to create/load streaming model")
        return False

    # Configure for low latency streaming
    model.set_streaming_parameters(
        chunk_size=1024,
        optimize_streaming_latency=3  # Maximum latency optimization
    )

    text = ("This is a longer text to demonstrate streaming synthesis. "
            "The audio should be generated in chunks as the text is processed. "
            "This allows for lower latency in real-time applications.")

    print(f"Streaming synthesis for: '{text[:50]}...'")

    # Create output file for streaming
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        output_path = temp_file.name

    try:
        # Get streaming generator
        audio_stream = model.synthesize_streaming(text)

        if audio_stream:
            print("✅ Streaming synthesis started...")

            # Collect chunks and save to file
            chunk_count = 0
            with open(output_path, 'wb') as f:
                for chunk in audio_stream:
                    if chunk:
                        f.write(chunk)
                        chunk_count += 1
                        if chunk_count % 10 == 0:  # Progress indicator
                            print(f"  Received {chunk_count} chunks...")

            print(f"✅ Streaming synthesis completed! Received {chunk_count} chunks")
            print(f"Audio saved to: {output_path}")

            # Check file size
            file_size = os.path.getsize(output_path)
            print(f"Final file size: {file_size} bytes")

        else:
            print("❌ Failed to get streaming audio")

    except Exception as e:
        print(f"❌ Streaming synthesis error: {e}")

    model.unload()
    return True


def model_info_example():
    """Display model information."""
    print("\n📊 Model Information Example")
    print("=" * 50)

    # Create model
    model = ModelFactory.create_tts_model(
        model_name="elevenlabs",
        device="cpu",
        api_key=os.getenv("ELEVENLABS_API_KEY")
    )

    if not model or not model.load():
        print("❌ Failed to create/load model")
        return False

    # Get model information
    info = model.get_model_info()

    print("Model Information:")
    for key, value in info.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")

    model.unload()
    return True


def main():
    """Run all examples."""
    print("🎤 ElevenLabs TTS Integration Examples")
    print("=" * 60)

    # Check API key
    if not check_api_key():
        return 1

    try:
        # Run examples
        examples = [
            ("Basic Synthesis", basic_synthesis_example),
            ("Voice Selection", voice_selection_example),
            ("Streaming Synthesis", streaming_synthesis_example),
            ("Model Information", model_info_example),
        ]

        results = []
        for name, example_func in examples:
            print(f"\n{'='*60}")
            print(f"Running: {name}")
            print(f"{'='*60}")

            try:
                success = example_func()
                results.append((name, success))

                if success:
                    print(f"✅ {name} completed successfully")
                else:
                    print(f"❌ {name} failed")

            except Exception as e:
                print(f"❌ {name} failed with error: {e}")
                results.append((name, False))

        # Summary
        print(f"\n{'='*60}")
        print("📋 SUMMARY")
        print(f"{'='*60}")

        for name, success in results:
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"  {name}: {status}")

        successful = sum(1 for _, success in results if success)
        total = len(results)
        print(f"\nResults: {successful}/{total} examples completed successfully")

        if successful == total:
            print("\n🎉 All examples completed successfully!")
            print("You can now use ElevenLabs TTS in your HoldTranscribe application.")
        else:
            print(f"\n⚠️  {total - successful} examples failed. Check the error messages above.")

        return 0 if successful == total else 1

    except KeyboardInterrupt:
        print("\n\n⏹️  Examples interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
