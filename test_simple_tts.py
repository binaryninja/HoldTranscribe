#!/usr/bin/env python3
"""
Simple test script for the working Kyutai TTS implementation.

This tests the CLI-based approach which should work reliably.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Test the simple TTS implementation."""
    print("🔊 Testing Simple Kyutai TTS Implementation")
    print("=" * 50)

    try:
        from holdtranscribe.tts.kyutai_simple import KyutaiSimpleTTS
        from holdtranscribe.utils import set_debug

        # Enable debug mode
        set_debug(True)

        # Create TTS model
        print("🤖 Creating simple TTS model...")
        tts = KyutaiSimpleTTS(device="cuda")

        print("✅ TTS model created")
        print(f"   Model info: {tts.get_model_info()}")

        # Test loading
        print("\n📥 Testing model loading...")
        if tts.load():
            print("✅ Model loaded successfully")
        else:
            print("❌ Model loading failed")
            return 1

        # Test synthesis to file
        print("\n💾 Testing file synthesis...")
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            test_file = tmp_file.name

        test_text = "This is a test of the simple TTS implementation."

        start_time = time.time()
        if tts.synthesize(test_text, test_file):
            synthesis_time = time.time() - start_time
            file_size = Path(test_file).stat().st_size
            print(f"✅ File synthesis successful!")
            print(f"   File: {test_file}")
            print(f"   Size: {file_size:,} bytes")
            print(f"   Time: {synthesis_time:.2f} seconds")

            # Clean up
            Path(test_file).unlink()
        else:
            print("❌ File synthesis failed")
            return 1

        # Test speaker playback
        print("\n🔊 Testing speaker playback...")
        speaker_text = "Hello! This should play through your speakers."

        start_time = time.time()
        try:
            tts.play_audio_to_speakers(speaker_text)
            playback_time = time.time() - start_time
            print(f"✅ Speaker playback successful in {playback_time:.2f} seconds")
        except Exception as e:
            print(f"❌ Speaker playback failed: {e}")
            return 1

        # Test repeated usage
        print("\n🔄 Testing repeated usage...")
        test_phrases = [
            "First test phrase.",
            "Second test phrase.",
            "Third and final test phrase."
        ]

        for i, phrase in enumerate(test_phrases, 1):
            print(f"   Test {i}: {phrase}")
            try:
                start_time = time.time()
                tts.play_audio_to_speakers(phrase)
                duration = time.time() - start_time
                print(f"   ✅ Completed in {duration:.2f}s")
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                return 1

        print("\n🎉 All tests passed!")
        print("✅ Simple TTS implementation is working correctly")

        # Test integration with model factory
        print("\n🏭 Testing model factory integration...")
        from holdtranscribe.models import ModelFactory

        factory_tts = ModelFactory.create_tts_model("kyutai", "cuda")
        if factory_tts and factory_tts.load():
            print("✅ Model factory integration working")
            test_text = "Factory integration test successful."
            factory_tts.play_audio_to_speakers(test_text)
            factory_tts.unload()
        else:
            print("❌ Model factory integration failed")
            return 1

        # Clean up
        tts.unload()
        print("\n🧹 Cleanup completed")

        return 0

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure TTS dependencies are installed:")
        print("   pip install -r requirements-tts.txt")
        return 1
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
