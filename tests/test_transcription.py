#!/usr/bin/env python3
"""
Comprehensive transcription test script for HoldTranscribe.
Tests both Voxtral and Whisper transcription pipelines with synthetic audio.
"""

import sys
import os
import time
import tempfile
import wave
import numpy as np

# Add the holdtranscribe module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'holdtranscribe'))

def create_test_audio(duration=3.0, sample_rate=16000, frequency=440):
    """Create synthetic audio data for testing."""
    print(f"Creating test audio: {duration}s at {sample_rate}Hz, {frequency}Hz tone")

    # Generate a sine wave
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t)

    # Add some noise to make it more realistic
    noise = np.random.normal(0, 0.1, audio_data.shape)
    audio_data = audio_data + noise

    # Convert to int16 format (like real audio input)
    audio_data = (audio_data * 32767).astype(np.int16)

    # Convert to bytes
    audio_bytes = audio_data.tobytes()

    print(f"✓ Generated {len(audio_bytes)} bytes of test audio")
    return audio_bytes

def create_speech_like_audio(duration=2.0, sample_rate=16000):
    """Create more speech-like audio data using multiple frequencies."""
    print(f"Creating speech-like test audio: {duration}s at {sample_rate}Hz")

    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # Create speech-like formants (multiple frequency components)
    formant1 = 0.3 * np.sin(2 * np.pi * 200 * t)  # Low frequency
    formant2 = 0.2 * np.sin(2 * np.pi * 800 * t)  # Mid frequency
    formant3 = 0.1 * np.sin(2 * np.pi * 2400 * t) # High frequency

    # Add some modulation to simulate speech patterns
    modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    audio_data = (formant1 + formant2 + formant3) * modulation

    # Add realistic noise
    noise = np.random.normal(0, 0.05, audio_data.shape)
    audio_data = audio_data + noise

    # Convert to int16 format
    audio_data = (audio_data * 32767).astype(np.int16)
    audio_bytes = audio_data.tobytes()

    print(f"✓ Generated {len(audio_bytes)} bytes of speech-like audio")
    return audio_bytes

def test_model_transcription(model_type, model, processor, audio_data, sample_rate=16000):
    """Test transcription with a specific model."""
    print(f"\n--- Testing {model_type.upper()} Transcription ---")

    start_time = time.time()

    try:
        if model_type == "voxtral":
            print("Using Voxtral transcription pipeline...")

            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_path = tmp.name
                with wave.open(tmp, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_data)

            # Use Voxtral conversation format
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio",
                            "path": wav_path,
                        },
                    ],
                }
            ]

            # Process with Voxtral
            inputs = processor.apply_chat_template(conversation)
            inputs = inputs.to("cuda", dtype=model.dtype)

            # Generate transcription
            import torch
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=500, temperature=0.0)
                decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
                text = decoded_outputs[0].strip()

            # Clean up
            os.remove(wav_path)

        else:  # whisper
            print("Using Whisper transcription pipeline...")

            # Create temporary WAV file for Whisper
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_path = tmp.name
                with wave.open(tmp, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_data)

            # Whisper transcription
            segments, info = model.transcribe(
                wav_path,
                beam_size=1,
                temperature=0.0,
                best_of=1
            )

            text_segments = []
            for segment in segments:
                text_segments.append(segment.text.strip())

            text = " ".join(text_segments).strip()

            # Clean up
            os.remove(wav_path)

        transcribe_time = time.time() - start_time

        print(f"✓ Transcription completed in {transcribe_time:.3f}s")
        print(f"✓ Result: '{text}'")

        if text:
            print(f"✓ Non-empty transcription received ({len(text)} characters)")
            return True
        else:
            print("⚠ Empty transcription received")
            return False

    except Exception as e:
        print(f"✗ Transcription failed: {e}")
        return False

def test_full_pipeline():
    """Test the complete transcription pipeline."""
    print("=" * 60)
    print("HOLDTRANSCRIBE TRANSCRIPTION PIPELINE TEST")
    print("=" * 60)

    # Import required modules
    try:
        from main import load_model, detect_device
        print("✓ Successfully imported HoldTranscribe modules")
    except ImportError as e:
        print(f"✗ Failed to import modules: {e}")
        return False

    # Detect device
    device = detect_device()
    print(f"✓ Using device: {device}")

    # Test different audio types
    test_cases = [
        ("Simple Tone", create_test_audio(duration=2.0, frequency=440)),
        ("Speech-like Audio", create_speech_like_audio(duration=3.0)),
        ("Short Audio", create_test_audio(duration=0.5, frequency=800)),
    ]

    # Models to test
    models_to_test = [
        ("mistralai/Voxtral-Mini-3B-2507", "Voxtral Mini"),
        ("tiny", "Whisper Tiny"),
    ]

    all_tests_passed = True

    for model_name, model_desc in models_to_test:
        print(f"\n{'='*20} Testing {model_desc} {'='*20}")

        try:
            # Load model
            print(f"Loading {model_desc}...")
            start_load = time.time()
            model_type, model, processor = load_model(model_name, device, fast_mode=True, beam_size=1)
            load_time = time.time() - start_load
            print(f"✓ {model_desc} loaded in {load_time:.2f}s")

            # Test with different audio samples
            for test_name, audio_data in test_cases:
                print(f"\n--- Testing {model_desc} with {test_name} ---")
                success = test_model_transcription(model_type, model, processor, audio_data)
                if not success:
                    all_tests_passed = False
                    print(f"⚠ {test_name} test failed for {model_desc}")

            # Clean up model to free memory
            del model
            if processor:
                del processor

            print(f"✓ {model_desc} testing completed")

        except Exception as e:
            print(f"✗ Failed to test {model_desc}: {e}")
            all_tests_passed = False
            continue

    # Summary
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("✓ ALL TRANSCRIPTION TESTS PASSED!")
        print("🚀 HoldTranscribe is ready for use!")
    else:
        print("⚠ SOME TESTS FAILED")
        print("Check the output above for details")
    print("=" * 60)

    return all_tests_passed

def test_audio_formats():
    """Test different audio format handling."""
    print("\n--- Testing Audio Format Handling ---")

    sample_rates = [8000, 16000, 22050, 44100]
    durations = [0.5, 1.0, 2.0, 5.0]

    for sr in sample_rates:
        for duration in durations:
            try:
                audio_data = create_test_audio(duration=duration, sample_rate=sr)
                print(f"✓ Generated audio: {sr}Hz, {duration}s, {len(audio_data)} bytes")
            except Exception as e:
                print(f"✗ Failed to generate {sr}Hz, {duration}s audio: {e}")
                return False

    print("✓ All audio format tests passed")
    return True

def benchmark_models():
    """Benchmark different models for performance comparison."""
    print("\n--- Model Performance Benchmark ---")

    try:
        from main import load_model, detect_device

        device = detect_device()
        test_audio = create_speech_like_audio(duration=3.0)

        models_to_benchmark = [
            ("tiny", "Whisper Tiny"),
            ("base", "Whisper Base"),
            ("mistralai/Voxtral-Mini-3B-2507", "Voxtral Mini"),
        ]

        results = []

        for model_name, model_desc in models_to_benchmark:
            print(f"\nBenchmarking {model_desc}...")

            try:
                # Time model loading
                start_load = time.time()
                model_type, model, processor = load_model(model_name, device, fast_mode=True, beam_size=1)
                load_time = time.time() - start_load

                # Time transcription (multiple runs for average)
                transcription_times = []
                for i in range(3):
                    start_transcribe = time.time()
                    success = test_model_transcription(model_type, model, processor, test_audio)
                    transcribe_time = time.time() - start_transcribe
                    if success:
                        transcription_times.append(transcribe_time)

                avg_transcribe_time = np.mean(transcription_times) if transcription_times else float('inf')

                results.append({
                    'model': model_desc,
                    'load_time': load_time,
                    'avg_transcribe_time': avg_transcribe_time,
                    'success': len(transcription_times) > 0
                })

                # Clean up
                del model
                if processor:
                    del processor

            except Exception as e:
                print(f"✗ Benchmark failed for {model_desc}: {e}")
                results.append({
                    'model': model_desc,
                    'load_time': float('inf'),
                    'avg_transcribe_time': float('inf'),
                    'success': False
                })

        # Print benchmark results
        print("\n" + "=" * 50)
        print("BENCHMARK RESULTS")
        print("=" * 50)
        print(f"{'Model':<20} {'Load Time':<12} {'Transcribe Time':<15} {'Status'}")
        print("-" * 50)

        for result in results:
            load_str = f"{result['load_time']:.2f}s" if result['load_time'] != float('inf') else "FAILED"
            transcribe_str = f"{result['avg_transcribe_time']:.3f}s" if result['avg_transcribe_time'] != float('inf') else "FAILED"
            status = "✓" if result['success'] else "✗"
            print(f"{result['model']:<20} {load_str:<12} {transcribe_str:<15} {status}")

        return True

    except Exception as e:
        print(f"✗ Benchmark failed: {e}")
        return False

def main():
    """Main test function."""
    print("HoldTranscribe Comprehensive Transcription Test")

    # Test audio format handling
    if not test_audio_formats():
        print("✗ Audio format tests failed")
        return False

    # Test full pipeline
    if not test_full_pipeline():
        print("✗ Pipeline tests failed")
        return False

    # Run benchmarks
    benchmark_models()

    print("\n🎉 All transcription tests completed!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
