#!/usr/bin/env python3
"""
ElevenLabs TTS Setup Script for HoldTranscribe

This script helps you configure ElevenLabs TTS integration with HoldTranscribe.
It will guide you through:
1. Installing required packages
2. Setting up your API key
3. Testing the connection
4. Selecting voices
5. Running example synthesis

Usage:
    python setup_elevenlabs.py
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def print_header():
    """Print setup script header."""
    print("🎤 ElevenLabs TTS Setup for HoldTranscribe")
    print("=" * 50)
    print("This script will help you set up ElevenLabs text-to-speech integration.")
    print()


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.8+ required.")
        return False

    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True


def check_package_installation():
    """Check if required packages are installed."""
    print("\n📦 Checking package installation...")

    packages_to_check = [
        ("elevenlabs", "elevenlabs"),
        ("requests", "requests"),
    ]

    missing_packages = []

    for package_name, import_name in packages_to_check:
        try:
            __import__(import_name)
            print(f"✅ {package_name} is installed")
        except ImportError:
            print(f"❌ {package_name} is missing")
            missing_packages.append(package_name)

    return missing_packages


def install_packages(missing_packages):
    """Install missing packages."""
    if not missing_packages:
        return True

    print(f"\n🔧 Installing missing packages: {', '.join(missing_packages)}")

    # Add elevenlabs version requirement
    install_list = []
    for package in missing_packages:
        if package == "elevenlabs":
            install_list.append("elevenlabs>=1.0.0")
        else:
            install_list.append(package)

    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install"
        ] + install_list)
        print("✅ Packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        print("Please install manually:")
        for package in install_list:
            print(f"  pip install {package}")
        return False


def get_api_key():
    """Get ElevenLabs API key from user or environment."""
    print("\n🔑 ElevenLabs API Key Setup")
    print("-" * 30)

    # Check if already set in environment
    existing_key = os.getenv("ELEVENLABS_API_KEY")
    if existing_key:
        print(f"✅ API key found in environment: {existing_key[:8]}..." + "*" * (len(existing_key) - 8))

        use_existing = input("Use existing API key? (Y/n): ").strip().lower()
        if use_existing in ['', 'y', 'yes']:
            return existing_key

    print("\nTo get your ElevenLabs API key:")
    print("1. Visit: https://elevenlabs.io/app/settings/api-keys")
    print("2. Sign up for an account (free tier available)")
    print("3. Generate an API key")
    print("4. Copy the key and paste it below")
    print()

    while True:
        api_key = input("Enter your ElevenLabs API key: ").strip()

        if not api_key:
            print("❌ API key cannot be empty")
            continue

        if len(api_key) < 10:
            print("❌ API key seems too short")
            continue

        # Basic validation - ElevenLabs keys typically start with certain patterns
        if not any(api_key.startswith(prefix) for prefix in ['sk-', 'el_', 'eleven_']):
            confirm = input("⚠️  API key format doesn't match expected pattern. Continue anyway? (y/N): ").strip().lower()
            if confirm not in ['y', 'yes']:
                continue

        return api_key


def test_api_connection(api_key):
    """Test connection to ElevenLabs API."""
    print("\n🌐 Testing API connection...")

    try:
        from elevenlabs import ElevenLabs

        client = ElevenLabs(api_key=api_key)
        user_info = client.user.get()

        print("✅ API connection successful!")
        print(f"📊 Account info:")
        print(f"   Subscription: {user_info.subscription.tier}")
        print(f"   Characters used: {user_info.subscription.character_count:,}")
        print(f"   Character limit: {user_info.subscription.character_limit:,}")

        remaining = user_info.subscription.character_limit - user_info.subscription.character_count
        print(f"   Characters remaining: {remaining:,}")

        return True

    except Exception as e:
        print(f"❌ API connection failed: {e}")
        print("\nPossible issues:")
        print("- Invalid API key")
        print("- Network connectivity problems")
        print("- ElevenLabs service issues")
        return False


def list_available_voices(api_key):
    """List available voices and let user select default."""
    print("\n🎭 Available Voices")
    print("-" * 20)

    try:
        from elevenlabs import ElevenLabs

        client = ElevenLabs(api_key=api_key)
        voices_response = client.voices.get_all()
        voices = voices_response.voices

        if not voices:
            print("❌ No voices available")
            return None

        print(f"Found {len(voices)} available voices:")
        print()

        # Display voices in a nice format
        for i, voice in enumerate(voices[:10]):  # Show first 10
            print(f"{i+1:2}. {voice.name}")
            print(f"     ID: {voice.voice_id}")
            if hasattr(voice, 'description') and voice.description:
                print(f"     Description: {voice.description}")
            if hasattr(voice, 'category'):
                print(f"     Category: {voice.category}")
            print()

        if len(voices) > 10:
            print(f"... and {len(voices) - 10} more voices")

        # Let user select a voice
        while True:
            try:
                choice = input(f"Select voice (1-{min(len(voices), 10)}) or press Enter for default: ").strip()

                if not choice:
                    # Use default voice
                    selected_voice = None
                    for voice in voices:
                        if voice.voice_id == "21m00Tcm4TlvDq8ikWAM":  # Rachel
                            selected_voice = voice
                            break

                    if not selected_voice:
                        selected_voice = voices[0]  # Fallback to first voice

                    print(f"Using default voice: {selected_voice.name}")
                    return selected_voice

                choice_idx = int(choice) - 1
                if 0 <= choice_idx < min(len(voices), 10):
                    selected_voice = voices[choice_idx]
                    print(f"Selected voice: {selected_voice.name}")
                    return selected_voice
                else:
                    print(f"❌ Please enter a number between 1 and {min(len(voices), 10)}")

            except ValueError:
                print("❌ Please enter a valid number")

    except Exception as e:
        print(f"❌ Failed to list voices: {e}")
        return None


def test_synthesis(api_key, voice=None):
    """Test text-to-speech synthesis."""
    print("\n🗣️  Testing Text-to-Speech Synthesis")
    print("-" * 35)

    try:
        # Add current directory to path to import holdtranscribe
        sys.path.insert(0, str(Path(__file__).parent))

        from holdtranscribe.models import ModelFactory

        # Create TTS model
        model = ModelFactory.create_tts_model(
            "elevenlabs",
            "cpu",
            api_key=api_key
        )

        if not model:
            print("❌ Failed to create TTS model")
            return False

        # Load model
        if not model.load():
            print("❌ Failed to load TTS model")
            return False

        # Set voice if specified
        if voice:
            model.set_voice_parameters(voice_id=voice.voice_id)
            print(f"Using voice: {voice.name}")

        # Test synthesis
        test_text = "Hello! This is a test of ElevenLabs text-to-speech integration with HoldTranscribe."
        output_file = "elevenlabs_test.mp3"

        print(f"Synthesizing: '{test_text}'")
        print(f"Output file: {output_file}")

        success = model.synthesize(test_text, output_file)

        if success:
            # Check if file was created and has content
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                file_size = os.path.getsize(output_file)
                print(f"✅ Synthesis successful! Generated {file_size} bytes")
                print(f"🎵 Audio saved to: {output_file}")
                print("You can play this file with any media player to test the audio quality.")

                # Offer to play the file
                try_play = input("\nTry to play the audio now? (Y/n): ").strip().lower()
                if try_play in ['', 'y', 'yes']:
                    play_audio_file(output_file)

            else:
                print("❌ Synthesis appeared to succeed but no audio file was created")
                success = False
        else:
            print("❌ Synthesis failed")

        model.unload()
        return success

    except ImportError as e:
        print(f"❌ Failed to import HoldTranscribe modules: {e}")
        print("Make sure you're running this script from the HoldTranscribe directory")
        return False
    except Exception as e:
        print(f"❌ Synthesis test failed: {e}")
        return False


def play_audio_file(file_path):
    """Try to play audio file using system default player."""
    try:
        if sys.platform.startswith('linux'):
            # Try common Linux audio players
            players = ['paplay', 'aplay', 'mpv', 'vlc', 'mplayer']
            for player in players:
                try:
                    subprocess.run([player, file_path], check=True,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    print(f"✅ Playing audio with {player}")
                    return
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            print("⚠️  Could not find audio player. Please play the file manually.")

        elif sys.platform == 'darwin':  # macOS
            subprocess.run(['afplay', file_path], check=True)
            print("✅ Playing audio with afplay")

        elif sys.platform.startswith('win'):  # Windows
            os.startfile(file_path)
            print("✅ Opening audio with default Windows player")

    except Exception as e:
        print(f"⚠️  Could not play audio automatically: {e}")
        print("Please play the file manually to test audio quality.")


def save_configuration(api_key, voice=None):
    """Save configuration for persistent use."""
    print("\n💾 Saving Configuration")
    print("-" * 22)

    save_config = input("Save API key to environment permanently? (Y/n): ").strip().lower()
    if save_config not in ['', 'y', 'yes']:
        print("⚠️  API key not saved. You'll need to set it manually:")
        print(f"   export ELEVENLABS_API_KEY='{api_key}'")
        return

    # Determine shell config file
    home = Path.home()
    shell_configs = [
        home / '.bashrc',
        home / '.zshrc',
        home / '.profile'
    ]

    config_file = None
    for config in shell_configs:
        if config.exists():
            config_file = config
            break

    if not config_file:
        config_file = home / '.bashrc'  # Default

    try:
        # Check if API key is already in config
        if config_file.exists():
            content = config_file.read_text()
            if 'ELEVENLABS_API_KEY' in content:
                print("⚠️  API key already exists in configuration file")
                overwrite = input("Overwrite existing configuration? (y/N): ").strip().lower()
                if overwrite not in ['y', 'yes']:
                    return

        # Append to config file
        with open(config_file, 'a') as f:
            f.write(f"\n# ElevenLabs API key for HoldTranscribe\n")
            f.write(f"export ELEVENLABS_API_KEY='{api_key}'\n")

        print(f"✅ API key saved to {config_file}")
        print("Restart your terminal or run 'source ~/.bashrc' to apply changes")

    except Exception as e:
        print(f"❌ Failed to save configuration: {e}")
        print("Please add this line to your shell configuration manually:")
        print(f"   export ELEVENLABS_API_KEY='{api_key}'")


def print_summary():
    """Print setup completion summary."""
    print("\n🎉 ElevenLabs TTS Setup Complete!")
    print("=" * 40)
    print("Your ElevenLabs TTS integration is now ready to use.")
    print()
    print("Next steps:")
    print("1. Run HoldTranscribe with TTS enabled:")
    print("   python -m holdtranscribe.main --tts")
    print()
    print("2. Use AI assistant with TTS:")
    print("   python -m holdtranscribe.main --model mistralai/Voxtral-Mini-3B-2507 --tts")
    print()
    print("3. Try different ElevenLabs models:")
    print("   --tts-model eleven_turbo_v2_5    (faster)")
    print("   --tts-model eleven_flash_v2_5     (fastest)")
    print("   --tts-model eleven_multilingual_v2 (multilingual)")
    print()
    print("4. Run the example script:")
    print("   python examples/elevenlabs_tts_example.py")
    print()
    print("For troubleshooting, see TTS_SETUP.md")


def main():
    """Main setup function."""
    print_header()

    # Step 1: Check Python version
    if not check_python_version():
        return 1

    # Step 2: Check package installation
    missing_packages = check_package_installation()

    if missing_packages:
        install = input(f"\nInstall missing packages? (Y/n): ").strip().lower()
        if install in ['', 'y', 'yes']:
            if not install_packages(missing_packages):
                return 1
        else:
            print("❌ Cannot continue without required packages")
            return 1

    # Step 3: Get API key
    api_key = get_api_key()
    if not api_key:
        print("❌ API key is required")
        return 1

    # Step 4: Test API connection
    if not test_api_connection(api_key):
        return 1

    # Step 5: List voices and let user select
    selected_voice = list_available_voices(api_key)

    # Step 6: Test synthesis
    if not test_synthesis(api_key, selected_voice):
        print("⚠️  Synthesis test failed, but API connection works")
        print("You may still be able to use ElevenLabs TTS")

    # Step 7: Save configuration
    save_configuration(api_key, selected_voice)

    # Step 8: Print summary
    print_summary()

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⏹️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during setup: {e}")
        sys.exit(1)
