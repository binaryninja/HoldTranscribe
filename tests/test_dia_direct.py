#!/usr/bin/env python3
"""
Test script using the exact correct DIA pattern.
"""

import torch
from transformers import AutoProcessor, DiaForConditionalGeneration

def test_dia_direct():
    """Test DIA model using the exact correct pattern."""
    print("🚀 Testing DIA model with correct pattern")
    print("=" * 50)

    # Set up device and model checkpoint
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    model_checkpoint = "nari-labs/Dia-1.6B-0626"

    print(f"Using device: {torch_device}")
    print(f"Model: {model_checkpoint}")

    # Test text with multiple speakers
    text = [
        "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."
    ]

    print(f"Input text: {text[0][:50]}...")

    try:
        print("\n📥 Loading processor...")
        processor = AutoProcessor.from_pretrained(model_checkpoint)
        print("✅ Processor loaded successfully")

        print("📥 Loading model...")
        model = DiaForConditionalGeneration.from_pretrained(model_checkpoint).to(torch_device)
        print("✅ Model loaded successfully")

        print("🔄 Processing input text...")
        inputs = processor(text=text, padding=True, return_tensors="pt").to(torch_device)
        print("✅ Text processed successfully")

        print("🎵 Generating audio...")
        outputs = model.generate(
            **inputs,
            max_new_tokens=3072,
            guidance_scale=3.0,
            temperature=1.8,
            top_p=0.90,
            top_k=45
        )
        print("✅ Audio generated successfully")

        print("🔄 Decoding outputs...")
        outputs = processor.batch_decode(outputs)
        print("✅ Outputs decoded successfully")

        print("💾 Saving audio...")
        processor.save_audio(outputs, "dia_test_direct.wav")
        print("✅ Audio saved to: dia_test_direct.wav")

        print("\n🎉 DIA test completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ DIA test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_streaming_chunks():
    """Test with multiple text chunks for streaming."""
    print("\n🚀 Testing DIA streaming chunks")
    print("=" * 50)

    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    model_checkpoint = "nari-labs/Dia-1.6B-0626"

    # Multiple chunks for streaming test
    text_chunks = [
        "[S1] Hello there! Welcome to our streaming demonstration.",
        "[S2] This is amazing! The voice quality is really impressive.",
        "[S1] I agree. Each chunk generates independently but maintains speaker consistency."
    ]

    try:
        print("📥 Loading model components...")
        processor = AutoProcessor.from_pretrained(model_checkpoint)
        model = DiaForConditionalGeneration.from_pretrained(model_checkpoint).to(torch_device)
        print("✅ Model loaded successfully")

        all_audio = []

        for i, chunk in enumerate(text_chunks, 1):
            print(f"\n🎵 Generating chunk {i}: '{chunk[:40]}...'")

            # Process chunk
            inputs = processor(text=[chunk], padding=True, return_tensors="pt").to(torch_device)

            # Generate audio
            outputs = model.generate(
                **inputs,
                max_new_tokens=3072,
                guidance_scale=3.0,
                temperature=1.8,
                top_p=0.90,
                top_k=45
            )

            # Decode
            decoded_outputs = processor.batch_decode(outputs)
            all_audio.extend(decoded_outputs)

            print(f"✅ Chunk {i} generated successfully")

        # Save individual chunks and create combined audio
        print("\n💾 Saving individual chunks...")
        chunk_files = []
        for i, audio in enumerate(all_audio):
            chunk_file = f"dia_chunk_{i+1}.wav"
            processor.save_audio([audio], chunk_file)
            chunk_files.append(chunk_file)
            print(f"✅ Chunk {i+1} saved to: {chunk_file}")

        # Also save first chunk as main example
        if all_audio:
            processor.save_audio([all_audio[0]], "dia_test_streaming.wav")
            print("✅ First chunk saved as: dia_test_streaming.wav")

        print(f"\n🎉 Streaming test completed! Generated {len(text_chunks)} chunks")
        return True

    except Exception as e:
        print(f"\n❌ Streaming test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 DIA Model Direct Test")
    print("=" * 60)

    # Test basic functionality
    success1 = test_dia_direct()

    # Test streaming chunks
    success2 = test_streaming_chunks()

    if success1 and success2:
        print("\n🎉 All tests passed!")
        print("\nGenerated files:")
        print("- dia_test_direct.wav")
        print("- dia_test_streaming.wav")
        print("- dia_chunk_1.wav, dia_chunk_2.wav, dia_chunk_3.wav")
    else:
        print("\n❌ Some tests failed")
