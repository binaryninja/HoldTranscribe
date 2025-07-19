#!/usr/bin/env python3
"""
Simple example of using Dia TTS for text-to-speech generation.

This demonstrates the basic TTS functionality that HoldTranscribe uses
for generating speech from AI assistant responses.

Usage:
    python examples/simple_tts.py
    python examples/simple_tts.py "Your custom text here"
"""

import sys
import time
import argparse

def generate_with_transformers(text, model_name="nari-labs/Dia-1.6B-0626", output_file="simple_output.mp3"):
    """Generate speech using Hugging Face Transformers implementation."""
    try:
        from transformers import AutoProcessor, DiaForConditionalGeneration
        import torch

        print(f"Loading Dia model: {model_name}")

        # Detect device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # Load model and processor
        processor = AutoProcessor.from_pretrained(model_name)
        model = DiaForConditionalGeneration.from_pretrained(model_name).to(device)

        print("Model loaded successfully!")

        # Format text for Dia (add speaker tag if not present)
        if not text.startswith("[S"):
            formatted_text = f"[S1] {text}"
        else:
            formatted_text = text

        print(f"Generating speech for: '{formatted_text}'")

        # Generate speech
        start_time = time.time()
        inputs = processor(text=[formatted_text], padding=True, return_tensors="pt").to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=3072,
            guidance_scale=3.0,
            temperature=1.8,
            top_p=0.90,
            top_k=45
        )

        # Decode and save audio
        outputs = processor.batch_decode(outputs)
        processor.save_audio(outputs, output_file)

        generation_time = time.time() - start_time
        print(f"✅ Speech generated in {generation_time:.2f} seconds")
        print(f"📁 Audio saved to: {output_file}")

        return True

    except ImportError as e:
        print(f"❌ Transformers implementation not available: {e}")
        print("💡 Install with: pip install git+https://github.com/huggingface/transformers.git")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def generate_with_native(text, model_name="nari-labs/Dia-1.6B-0626", output_file="simple_output_native.mp3"):
    """Generate speech using native Dia implementation."""
    try:
        from dia.model import Dia

        print(f"Loading Dia model: {model_name}")

        # Load model
        model = Dia.from_pretrained(model_name, compute_dtype="float16")

        print("Model loaded successfully!")

        # Format text for Dia (add speaker tag if not present)
        if not text.startswith("[S"):
            formatted_text = f"[S1] {text}"
        else:
            formatted_text = text

        print(f"Generating speech for: '{formatted_text}'")

        # Generate speech
        start_time = time.time()
        output = model.generate(
            formatted_text,
            use_torch_compile=False,
            verbose=True,
            cfg_scale=3.0,
            temperature=1.8,
            top_p=0.90,
            cfg_filter_top_k=50,
        )

        # Save audio
        model.save_audio(output_file, output)

        generation_time = time.time() - start_time
        print(f"✅ Speech generated in {generation_time:.2f} seconds")
        print(f"📁 Audio saved to: {output_file}")

        return True

    except ImportError as e:
        print(f"❌ Native Dia implementation not available: {e}")
        print("💡 Install with: pip install git+https://github.com/nari-labs/dia.git")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Simple Dia TTS example")
    parser.add_argument("text", nargs="?",
                       default="Hello! This is a demonstration of the Dia text-to-speech system. It can generate natural sounding speech from text.",
                       help="Text to convert to speech")
    parser.add_argument("--model", default="nari-labs/Dia-1.6B-0626",
                       help="Dia model to use")
    parser.add_argument("--method", choices=["transformers", "native", "auto"], default="auto",
                       help="Which implementation to use")

    args = parser.parse_args()

    print("🎙️  Simple Dia TTS Example")
    print("=" * 50)
    print(f"📝 Text: {args.text}")
    print(f"🎯 Model: {args.model}")
    print(f"🔧 Method: {args.method}")
    print()

    success = False

    if args.method == "transformers":
        success = generate_with_transformers(args.text, args.model)
    elif args.method == "native":
        success = generate_with_native(args.text, args.model)
    else:  # auto - try transformers first, then native
        print("🔄 Trying transformers implementation first...")
        success = generate_with_transformers(args.text, args.model)

        if not success:
            print("\n🔄 Trying native implementation...")
            success = generate_with_native(args.text, args.model)

    if success:
        print("\n🎵 Success! Check the generated audio file(s)")
        print("💡 You can play the audio with your system's default audio player")
    else:
        print("\n❌ Failed to generate speech")
        print("💡 Make sure you have Dia installed:")
        print("   pip install git+https://github.com/nari-labs/dia.git")
        print("   # or for transformers support:")
        print("   pip install git+https://github.com/huggingface/transformers.git")
        sys.exit(1)

if __name__ == "__main__":
    main()
