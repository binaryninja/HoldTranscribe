#!/usr/bin/env python3
"""
Moshi TTS Demo for HoldTranscribe

This script demonstrates the new Moshi TTS functionality that is now the default
TTS implementation in HoldTranscribe. It showcases key features including:

- Basic text-to-speech synthesis
- Real-time streaming generation
- Voice parameter adjustment
- Model information display

Usage:
    python demo_moshi_tts.py
    python demo_moshi_tts.py --cpu          # Force CPU usage
    python demo_moshi_tts.py --quick        # Quick demo mode
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

def print_header():
    """Print demo header."""
    print("🎙️  Moshi TTS Demo - HoldTranscribe")
    print("=" * 50)
    print("Demonstrating Kyutai's Moshi model for text-to-speech")
    print("Now the default TTS implementation in HoldTranscribe")
    print()

def check_dependencies():
    """Check if required dependencies are available."""
    print("🔍 Checking dependencies...")

    missing_deps = []

    try:
        import torch
        print(f"   ✓ PyTorch {torch.__version__}")

        if torch.cuda.is_available():
            print(f"   ✓ CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("   ℹ CUDA not available, will use CPU")
    except ImportError:
        missing_deps.append("torch")

    try:
        import transformers
        print(f"   ✓ Transformers {transformers.__version__}")
    except ImportError:
        missing_deps.append("transformers")

    try:
        import soundfile
        print(f"   ✓ SoundFile available")
    except ImportError:
        missing_deps.append("soundfile")

    try:
        import numpy
        print(f"   ✓ NumPy {numpy.__version__}")
    except ImportError:
        missing_deps.append("numpy")

    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install -r requirements-tts.txt")
        return False

    print("   ✅ All dependencies available")
    return True

def demo_basic_synthesis(model, quick_mode=False):
    """Demonstrate basic text-to-speech synthesis."""
    print("\n📝 Demo 1: Basic Text-to-Speech Synthesis")
    print("-" * 40)

    if quick_mode:
        test_text = "Hello from Moshi!"
    else:
        test_text = input("Enter text to synthesize (or press Enter for default): ").strip()
        if not test_text:
            test_text = "Hello! This is a demonstration of Moshi text-to-speech. The quality is quite impressive!"

    print(f"Text: '{test_text}'")

    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        output_file = tmp.name

    try:
        print("Generating audio...")
        start_time = time.time()

        success = model.synthesize(test_text, output_file)

        generation_time = time.time() - start_time

        if success and os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"✅ Success! Generated in {generation_time:.1f}s")
            print(f"   File: {output_file}")
            print(f"   Size: {file_size:,} bytes")

            if not quick_mode:
                play_choice = input("Play audio? (y/n): ").lower()
                if play_choice == 'y':
                    try:
                        print("Playing audio...")
                        model.play_audio_to_speakers(test_text)
                        print("✅ Playback completed")
                    except Exception as e:
                        print(f"⚠ Playback failed: {e}")
                        print(f"You can manually play: {output_file}")
        else:
            print("❌ Synthesis failed")
            return False

    except Exception as e:
        print(f"❌ Error during synthesis: {e}")
        return False

    finally:
        # Cleanup
        try:
            if os.path.exists(output_file):
                os.unlink(output_file)
        except:
            pass

    return True

def demo_streaming_synthesis(model, quick_mode=False):
    """Demonstrate streaming text-to-speech synthesis."""
    print("\n🌊 Demo 2: Streaming Synthesis")
    print("-" * 40)

    if quick_mode:
        long_text = "This is a streaming test. Each chunk is generated separately for real-time applications."
    else:
        long_text = input("Enter long text for streaming (or press Enter for default): ").strip()
        if not long_text:
            long_text = (
                "This is a demonstration of streaming text-to-speech synthesis. "
                "The text is split into chunks and each chunk is generated separately. "
                "This allows for real-time applications where you want to start playing "
                "audio before the entire text is processed. It's particularly useful "
                "for conversational AI applications and live speech generation."
            )

    print(f"Text: '{long_text[:100]}{'...' if len(long_text) > 100 else ''}'")

    try:
        print("Generating streaming audio...")

        chunks_generated = 0
        total_samples = 0
        start_time = time.time()

        for audio_chunk in model.synthesize_streaming(long_text):
            chunks_generated += 1
            total_samples += len(audio_chunk)
            print(f"   Chunk {chunks_generated}: {len(audio_chunk):,} samples")

        streaming_time = time.time() - start_time

        print(f"✅ Streaming complete!")
        print(f"   Total chunks: {chunks_generated}")
        print(f"   Total samples: {total_samples:,}")
        print(f"   Time: {streaming_time:.1f}s")

        return True

    except Exception as e:
        print(f"❌ Streaming failed: {e}")
        return False

def demo_voice_parameters(model):
    """Demonstrate voice parameter adjustment."""
    print("\n🎛️  Demo 3: Voice Parameter Adjustment")
    print("-" * 40)

    try:
        # Show current parameters
        info = model.get_model_info()
        print("Current parameters:")
        print(f"   Temperature: {info.get('temperature', 'N/A')}")
        print(f"   Max tokens: {info.get('max_new_tokens', 'N/A')}")

        # Test different temperatures
        test_text = "The weather is nice today."
        temperatures = [0.3, 0.6, 0.9]

        print(f"\nTesting different temperatures with: '{test_text}'")

        for temp in temperatures:
            print(f"\n   Testing temperature {temp}...")
            model.set_voice_parameters(temperature=temp, max_new_tokens=30)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                output_file = tmp.name

            try:
                start_time = time.time()
                success = model.synthesize(test_text, output_file)
                gen_time = time.time() - start_time

                if success:
                    file_size = os.path.getsize(output_file)
                    print(f"      ✅ Generated in {gen_time:.1f}s ({file_size:,} bytes)")
                else:
                    print(f"      ❌ Failed")
            finally:
                try:
                    os.unlink(output_file)
                except:
                    pass

        # Reset to defaults
        model.set_voice_parameters(temperature=0.6, max_new_tokens=50)
        print("\n   Reset to default parameters")

        return True

    except Exception as e:
        print(f"❌ Parameter demo failed: {e}")
        return False

def demo_model_info(model):
    """Display comprehensive model information."""
    print("\n🔍 Demo 4: Model Information")
    print("-" * 40)

    try:
        info = model.get_model_info()

        print("Model Details:")
        for key, value in info.items():
            print(f"   {key.replace('_', ' ').title()}: {value}")

        # Available voices
        voices = model.get_available_voices()
        print(f"\nAvailable Voices: {voices}")

        return True

    except Exception as e:
        print(f"❌ Info demo failed: {e}")
        return False

def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Moshi TTS Demo")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--quick", action="store_true", help="Quick demo mode (no user input)")
    parser.add_argument("--model", default="default", help="Model to use")

    args = parser.parse_args()

    print_header()

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Set device
    device = "cpu" if args.cpu else ("cuda" if check_cuda() else "cpu")
    print(f"\n🖥️  Using device: {device}")

    # Load model
    print(f"\n📦 Loading Moshi TTS model: {args.model}")

    try:
        from holdtranscribe.models import ModelFactory

        model = ModelFactory.create_tts_model(args.model, device)

        if not model:
            print("❌ Failed to create model")
            sys.exit(1)

        print("Loading model...")
        start_time = time.time()

        if not model.load():
            print("❌ Failed to load model")
            sys.exit(1)

        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.1f}s")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure HoldTranscribe is properly installed")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        sys.exit(1)

    # Run demos
    demos = [
        ("Basic Synthesis", lambda: demo_basic_synthesis(model, args.quick)),
        ("Streaming Synthesis", lambda: demo_streaming_synthesis(model, args.quick)),
        ("Voice Parameters", lambda: demo_voice_parameters(model)),
        ("Model Information", lambda: demo_model_info(model))
    ]

    if args.quick:
        print("\n🚀 Running quick demo mode...")

    results = []
    for demo_name, demo_func in demos:
        try:
            if not args.quick:
                input(f"\nPress Enter to run '{demo_name}' demo...")

            result = demo_func()
            results.append((demo_name, result))

        except KeyboardInterrupt:
            print("\n⚠ Demo interrupted by user")
            break
        except Exception as e:
            print(f"❌ Demo '{demo_name}' crashed: {e}")
            results.append((demo_name, False))

    # Cleanup
    try:
        model.unload()
        print("\n🧹 Model unloaded")
    except:
        pass

    # Summary
    print("\n" + "=" * 50)
    print("📊 DEMO SUMMARY")
    print("=" * 50)

    passed = 0
    for demo_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {demo_name}")
        if result:
            passed += 1

    total = len(results)
    print(f"\nResults: {passed}/{total} demos successful")

    if passed == total:
        print("\n🎉 All demos completed successfully!")
        print("Moshi TTS is working perfectly and ready to use in HoldTranscribe!")
    else:
        print("\n⚠ Some demos failed. Check the output above for details.")

    print("\n💡 To use Moshi TTS in HoldTranscribe:")
    print("   python -m holdtranscribe.main --tts")
    print("   (Moshi is now the default TTS model)")

def check_cuda():
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False

if __name__ == "__main__":
    main()
