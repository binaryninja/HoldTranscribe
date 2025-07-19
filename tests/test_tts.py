#!/usr/bin/env python3
"""
Test script for Dia TTS integration in HoldTranscribe.

This script tests the text-to-speech functionality using the Dia model
to ensure it works correctly before integrating with the main application.

Usage:
    python examples/test_tts.py
    python examples/test_tts.py --text "Hello, this is a test"
    python examples/test_tts.py --model "nari-labs/Dia-1.6B-0626"
"""

import argparse
import os
import sys
import time

# Add parent directory to path so we can import from holdtranscribe
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_dia_transformers(model_name: str, device: str, text: str, output_file: str):
    """Test Dia using Hugging Face Transformers implementation."""
    try:
        from transformers import AutoProcessor, DiaForConditionalGeneration

        print(f"Loading Dia model '{model_name}' using transformers...")
        processor = AutoProcessor.from_pretrained(model_name)
        model = DiaForConditionalGeneration.from_pretrained(model_name).to(device)

        print(f"Model loaded successfully on {device}")

        # Format text for Dia
        if not text.startswith("[S"):
            formatted_text = f"[S1] {text}"
        else:
            formatted_text = text

        print(f"Generating speech for: '{formatted_text}'")

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

        outputs = processor.batch_decode(outputs)
        processor.save_audio(outputs, output_file)

        generation_time = time.time() - start_time
        print(f"✅ Speech generated successfully in {generation_time:.2f}s")
        print(f"   Output saved to: {output_file}")
        return True

    except ImportError as e:
        print(f"❌ Transformers implementation not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Error with transformers implementation: {e}")
        return False

def test_dia_native(model_name: str, text: str, output_file: str):
    """Test Dia using native implementation."""
    try:
        from dia.model import Dia

        print(f"Loading Dia model '{model_name}' using native implementation...")
        model = Dia.from_pretrained(model_name, compute_dtype="float16")

        print("Model loaded successfully")

        # Format text for Dia
        if not text.startswith("[S"):
            formatted_text = f"[S1] {text}"
        else:
            formatted_text = text

        print(f"Generating speech for: '{formatted_text}'")

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

        model.save_audio(output_file, output)

        generation_time = time.time() - start_time
        print(f"✅ Speech generated successfully in {generation_time:.2f}s")
        print(f"   Output saved to: {output_file}")
        return True

    except ImportError as e:
        print(f"❌ Native Dia implementation not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Error with native implementation: {e}")
        return False

def detect_device():
    """Detect available device (CUDA/CPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"

def main():
    parser = argparse.ArgumentParser(description="Test Dia TTS integration")
    parser.add_argument("--text", default="Hello! This is a test of the Dia text to speech system. It should generate natural sounding speech.",
                       help="Text to convert to speech")
    parser.add_argument("--model", default="nari-labs/Dia-1.6B-0626",
                       help="Dia model to use")
    parser.add_argument("--output", default="test_output.mp3",
                       help="Output audio file")
    parser.add_argument("--method", choices=["transformers", "native", "both"], default="both",
                       help="Which implementation to test")

    args = parser.parse_args()

    device = detect_device()
    print(f"🖥️  Using device: {device}")
    print(f"📝 Text: {args.text}")
    print(f"🎯 Model: {args.model}")
    print(f"📁 Output: {args.output}")
    print("=" * 60)

    success_count = 0
    total_tests = 0

    if args.method in ["transformers", "both"]:
        print("\n🧪 Testing Transformers implementation...")
        total_tests += 1
        if test_dia_transformers(args.model, device, args.text, f"transformers_{args.output}"):
            success_count += 1

    if args.method in ["native", "both"]:
        print("\n🧪 Testing Native implementation...")
        total_tests += 1
        if test_dia_native(args.model, args.text, f"native_{args.output}"):
            success_count += 1

    print("\n" + "=" * 60)
    print(f"📊 Test Results: {success_count}/{total_tests} implementations working")

    if success_count == 0:
        print("❌ No Dia implementations are working. Please install Dia:")
        print("   pip install git+https://github.com/nari-labs/dia.git")
        print("   # or for transformers support:")
        print("   pip install git+https://github.com/huggingface/transformers.git")
        sys.exit(1)
    elif success_count < total_tests:
        print("⚠️  Some implementations failed, but at least one is working")
    else:
        print("✅ All tested implementations are working!")

    print("\n🎵 Check the generated audio files to verify quality")

if __name__ == "__main__":
    main()
