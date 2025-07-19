#!/usr/bin/env python3
"""
Test script for the full AI Assistant TTS pipeline in HoldTranscribe.

This script simulates what happens when a user uses the AI assistant hotkey
with TTS enabled, testing the complete flow from AI response to audio playback.

Usage:
    python test_assistant_pipeline.py
    python test_assistant_pipeline.py --text "What is the weather like today?"
    python test_assistant_pipeline.py --response "Custom AI response text"
"""

import sys
import os
import time
import argparse
import platform
import subprocess
import uuid

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def simulate_ai_response(user_question):
    """Simulate a realistic AI assistant response."""
    responses = {
        "what is the weather like today": "I don't have access to real-time weather data, but you can check the current weather by asking your voice assistant, checking a weather app, or looking online at sites like Weather.com or AccuWeather.",

        "how do i cook pasta": "To cook pasta, bring a large pot of salted water to a boil. Add the pasta and cook according to package directions, usually 8-12 minutes. Stir occasionally to prevent sticking. Test for doneness by tasting. Drain and serve immediately with your favorite sauce.",

        "what time is it": "I don't have access to the current time, but you can check the time on your computer's clock, phone, or by asking your system's voice assistant.",

        "tell me a joke": "Why don't scientists trust atoms? Because they make up everything! Here's another one: Why did the scarecrow win an award? Because he was outstanding in his field!",

        "explain quantum computing": "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information. Unlike classical computers that use bits (0 or 1), quantum computers use quantum bits or qubits that can be in multiple states simultaneously, potentially solving certain problems much faster than classical computers.",

        "default": "I'm an AI assistant integrated with HoldTranscribe. I can help answer questions, provide information, and have conversations. What would you like to know about?"
    }

    # Try to find a matching response
    question_lower = user_question.lower().strip()
    for key, response in responses.items():
        if key in question_lower:
            return response

    return responses["default"]

def test_assistant_pipeline(user_input, ai_response, enable_playback=True):
    """Test the full AI assistant TTS pipeline."""

    print("🤖 AI Assistant Pipeline Test")
    print("=" * 50)
    print(f"👤 User: {user_input}")
    print(f"🤖 AI Response: {ai_response}")
    print()

    # Import HoldTranscribe modules
    try:
        import holdtranscribe.main
        import sys
        ht_main = sys.modules['holdtranscribe.main']

        # Set up globals for TTS
        ht_main.DEBUG = True
        ht_main.DEVICE = "cuda"
        ht_main.ENABLE_TTS = True

        print("📝 Step 1: Copying AI response to clipboard...")
        try:
            import pyperclip
            pyperclip.copy(ai_response)
            print("✅ Text copied to clipboard")
        except Exception as e:
            print(f"⚠️  Clipboard copy failed: {e}")

        print("\n🎵 Step 2: Loading Dia TTS model...")
        start_time = time.time()

        # Load TTS model
        dia_model, dia_processor = ht_main.load_dia_model("nari-labs/Dia-1.6B-0626", "cuda")

        if dia_model is None:
            print("❌ Failed to load Dia model")
            return False

        load_time = time.time() - start_time
        print(f"✅ TTS model loaded in {load_time:.2f}s")

        # Set global variables for generate_speech
        ht_main.dia_model = dia_model
        ht_main.dia_processor = dia_processor

        print("\n🔊 Step 3: Generating speech from AI response...")
        tts_start = time.time()

        # Generate unique filename like the real pipeline does
        tts_filename = f"assistant_response_{uuid.uuid4().hex[:8]}.mp3"

        success = ht_main.generate_speech(ai_response, tts_filename)

        if not success:
            print("❌ TTS generation failed")
            return False

        tts_time = time.time() - tts_start
        print(f"✅ TTS generated in {tts_time:.2f}s")

        # Check file was created
        if os.path.exists(tts_filename):
            file_size = os.path.getsize(tts_filename)
            print(f"📁 Audio file created: {tts_filename} ({file_size:,} bytes)")
        else:
            print("❌ Audio file was not created")
            return False

        print("\n🎤 Step 4: Attempting auto-playback...")
        if enable_playback:
            try:
                # Same auto-play logic as the real pipeline
                if platform.system() == "Linux":
                    result = subprocess.run(["xdg-open", tts_filename], check=False,
                                          capture_output=True, text=True)
                elif platform.system() == "Darwin":  # macOS
                    result = subprocess.run(["open", tts_filename], check=False,
                                          capture_output=True, text=True)
                elif platform.system() == "Windows":
                    result = subprocess.run(["start", tts_filename], shell=True, check=False,
                                          capture_output=True, text=True)
                else:
                    print(f"⚠️  Unknown platform: {platform.system()}")
                    return True

                if result.returncode == 0:
                    print(f"✅ Audio playback initiated using system default player")
                    print(f"🎵 Playing: {tts_filename}")
                else:
                    print(f"⚠️  Playback command returned code {result.returncode}")
                    if result.stderr:
                        print(f"   Error: {result.stderr.strip()}")

            except Exception as e:
                print(f"⚠️  Could not auto-play audio: {e}")
        else:
            print("⏭️  Auto-playback skipped (disabled)")

        print("\n📱 Step 5: Simulating notification...")
        try:
            import notify2
            notify2.init("AI Assistant Test")
            truncated_text = ai_response[:120] + ("…" if len(ai_response) > 120 else "")
            notification = notify2.Notification("🤖 AI Assistant Response 🔊", truncated_text)
            notification.show()
            print("✅ Desktop notification shown")
        except Exception as e:
            print(f"⚠️  Notification failed: {e}")

        print("\n🎉 Pipeline test completed successfully!")
        print(f"📊 Summary:")
        print(f"   Model load time: {load_time:.2f}s")
        print(f"   TTS generation time: {tts_time:.2f}s")
        print(f"   Total pipeline time: {time.time() - start_time:.2f}s")
        print(f"   Audio file: {tts_filename}")

        return True

    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Test AI Assistant TTS Pipeline")
    parser.add_argument("--text", default="What is the weather like today?",
                       help="Simulated user input")
    parser.add_argument("--response", default=None,
                       help="Custom AI response (if not provided, will be generated)")
    parser.add_argument("--no-playback", action="store_true",
                       help="Skip audio playback attempt")

    args = parser.parse_args()

    # Generate or use provided AI response
    if args.response:
        ai_response = args.response
    else:
        ai_response = simulate_ai_response(args.text)

    # Run the pipeline test
    success = test_assistant_pipeline(
        user_input=args.text,
        ai_response=ai_response,
        enable_playback=not args.no_playback
    )

    if success:
        print("\n✨ The AI Assistant TTS pipeline is working correctly!")
        print("💡 In the real application, this would happen when you:")
        print("   1. Hold Ctrl/Cmd + Mouse Back Button")
        print("   2. Speak your question")
        print("   3. Release the hotkey")
        print("   4. AI processes and responds with both text and speech")
    else:
        print("\n❌ Pipeline test failed - check error messages above")
        sys.exit(1)

if __name__ == "__main__":
    main()
