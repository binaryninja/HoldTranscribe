# HoldTranscribe

Hotkey-Activated Voice-to-Clipboard Transcriber

A lightweight tool that records audio while you hold a configurable hotkey, transcribes speech using OpenAI's Whisper model, and copies the result to your clipboard.

---

## Features

* Hold-to-record using a customizable hotkey combination
* GPU acceleration with automatic CUDA detection and CPU fallback
* Instant copy of transcribed text to the clipboard
* **AI Assistant Mode with Voxtral** for intelligent voice interactions
* **Text-to-Speech (TTS) output** using ElevenLabs for assistant responses
* Persistent model instance for low-latency transcription
* Configurable model size and beam search settings
* Detailed debug output and performance metrics
* Cross-platform support (Linux, macOS, Windows)
* Voice Activity Detection (VAD) for clean audio capture
* Auto-start service integration for all platforms
* Dual hotkey system: separate keys for transcription and AI assistant

---

## Platform-Specific Requirements

### Linux
* Python 3.8 or later
* Bash-compatible shell (for installer script)
* A CUDA-capable GPU (optional, for hardware acceleration)
* PulseAudio or equivalent audio system
* Permissions to read input events (user in `input` group)
* X11 or Wayland desktop environment

### macOS
* Python 3.8 or later
* macOS 10.14 (Mojave) or later
* Microphone access permissions
* Accessibility permissions for global hotkey monitoring
* Optional: CUDA-capable GPU (limited support on newer Macs)

### Windows
* Python 3.8 or later
* Windows 10 or later (Windows 11 recommended)
* Microphone access permissions
* Optional: CUDA-capable GPU with appropriate drivers
* PowerShell 5.0 or later (for service installation)

---

## Installation

### Option 1: Pip Installation (Recommended)

**From GitHub (all platforms):**
```bash
pip install git+https://github.com/binaryninja/holdtranscribe.git
```

**From PyPI (when available):**
```bash
pip install holdtranscribe
```

### Option 2: Manual Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/binaryninja/holdtranscribe.git
   cd holdtranscribe
   ```

2. **Install Python dependencies:**
   ```bash
   pip install faster-whisper sounddevice pynput webrtcvad pyperclip notify2 numpy psutil
   
   # For AI Assistant support
   pip install git+https://github.com/huggingface/transformers
   pip install mistral-common[audio]
   
   # For Text-to-Speech support (ElevenLabs - default)
   pip install elevenlabs>=1.0.0
   
   # Alternative TTS models
   pip install git+https://github.com/nari-labs/dia.git  # For DIA TTS
   pip install moshi>=0.2.10 sphn>=0.1.0  # For Moshi TTS
   ```

3. **Optional GPU acceleration:**
   
   **Linux/Windows with CUDA:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
   
   **macOS with Metal Performance Shaders:**
   ```bash
   pip install torch torchvision torchaudio
   ```

---

## Platform-Specific Setup

### Linux Setup

1. **Add user to input group (if needed):**
   ```bash
   sudo usermod -aG input $USER
   ```
   Log out and back in for changes to take effect.

2. **Install system dependencies (Ubuntu/Debian):**
   ```bash
   sudo apt update
   sudo apt install python3-pip portaudio19-dev pulseaudio
   ```

3. **Install system dependencies (Fedora/RHEL):**
   ```bash
   sudo dnf install python3-pip portaudio-devel pulseaudio
   ```

### macOS Setup

1. **Install dependencies via Homebrew:**
   ```bash
   # Install Homebrew if not already installed
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   
   # Install PortAudio
   brew install portaudio
   ```

2. **Grant permissions:**
   - **Microphone Access:** System Preferences → Security & Privacy → Privacy → Microphone → Enable for Terminal/your Python environment
   - **Accessibility Access:** System Preferences → Security & Privacy → Privacy → Accessibility → Enable for Terminal/your Python environment
   - **Input Monitoring:** System Preferences → Security & Privacy → Privacy → Input Monitoring → Enable for Terminal/your Python environment

3. **For Apple Silicon Macs:**
   ```bash
   # Install Python dependencies with conda for better compatibility
   conda install python=3.9
   pip install faster-whisper sounddevice pynput webrtcvad pyperclip notify2 numpy psutil
   ```

### Windows Setup

1. **Install via Microsoft Store or python.org:**
   - Download Python from [python.org](https://python.org) or install via Microsoft Store
   - Ensure "Add Python to PATH" is checked during installation

2. **Install Visual C++ Build Tools (if compilation errors occur):**
   - Download and install Microsoft C++ Build Tools
   - Or install Visual Studio Community with C++ workload

3. **Grant microphone permissions:**
   - Settings → Privacy → Microphone → Allow apps to access microphone → Enable for Python/Terminal

---

## Usage

### Basic Usage (All Platforms)

```bash
# Run with default settings (if installed via pip)
holdtranscribe

# Or if using the script directly
python voice_hold_to_clip.py
```

### Command Line Options

```bash
--model <size>       Model to use: Whisper (tiny, base, small, medium, large-v3) or 
                     Voxtral (mistralai/Voxtral-Mini-3B-2507). Default: large-v3
--beam-size <n>      Beam search width for Whisper (1 for fastest). Default: 5
--fast               Shorthand for `--model base --beam-size 1`
--debug              Enable verbose timing and resource metrics
--device <cpu|cuda>  Force CPU or GPU mode
--tts                Enable text-to-speech output for AI assistant responses
--tts-model <model>  TTS model to use. Default: ElevenLabs (eleven_multilingual_v2)
                     Options: elevenlabs, eleven_turbo_v2_5, moshi, dia models
--tts-output <file>  Output file for TTS audio. Default: assistant_response.mp3
```

### Hotkey Controls

The application supports dual hotkey operation:

**Linux/Windows:**
- **Ctrl + Mouse Button 9 (Forward)**: Transcription mode
- **Ctrl + Mouse Button 8 (Back)**: AI Assistant mode

**macOS:**
- **Cmd + Mouse Button 9 (Forward)**: Transcription mode  
- **Cmd + Mouse Button 8 (Back)**: AI Assistant mode

### AI Assistant Mode with TTS

Enable intelligent voice interactions with text-to-speech responses:

```bash
# Basic AI assistant with Voxtral
holdtranscribe --model mistralai/Voxtral-Mini-3B-2507

# AI assistant with TTS enabled
holdtranscribe --model mistralai/Voxtral-Mini-3B-2507 --tts

# Custom TTS model (alternative models)
holdtranscribe --model mistralai/Voxtral-Mini-3B-2507 --tts --tts-model moshi
holdtranscribe --model mistralai/Voxtral-Mini-3B-2507 --tts --tts-model dia
```

### ElevenLabs TTS Setup (Default)

HoldTranscribe now uses ElevenLabs as the default TTS provider for high-quality voice synthesis.

#### Getting Started

1. **Get an API Key:**
   - Visit [ElevenLabs](https://elevenlabs.io/app/settings/api-keys)
   - Sign up for an account (free tier available with 10,000 characters/month)
   - Generate an API key

2. **Set Your API Key:**
   ```bash
   # Linux/macOS
   export ELEVENLABS_API_KEY="your_api_key_here"
   
   # Windows Command Prompt
   set ELEVENLABS_API_KEY=your_api_key_here
   
   # Windows PowerShell
   $env:ELEVENLABS_API_KEY="your_api_key_here"
   ```

3. **Test ElevenLabs Integration:**
   ```bash
   # Run the integration test
   python test_elevenlabs_integration.py
   
   # Run the example script
   python examples/elevenlabs_tts_example.py
   ```

#### ElevenLabs Model Options

```bash
# Default high-quality model
holdtranscribe --model mistralai/Voxtral-Mini-3B-2507 --tts

# Fast synthesis (lower latency)
holdtranscribe --model mistralai/Voxtral-Mini-3B-2507 --tts --tts-model eleven_turbo_v2_5

# Fastest synthesis (minimal latency)
holdtranscribe --model mistralai/Voxtral-Mini-3B-2507 --tts --tts-model eleven_flash_v2_5

# Multilingual support
holdtranscribe --model mistralai/Voxtral-Mini-3B-2507 --tts --tts-model eleven_multilingual_v2
```

#### Voice Selection

ElevenLabs offers various voices. You can customize voice settings programmatically:

```python
from holdtranscribe.models import ModelFactory
import os

# Create TTS model with custom voice
tts_model = ModelFactory.create_tts_model(
    "elevenlabs", 
    "cpu",
    api_key=os.getenv("ELEVENLABS_API_KEY")
)

if tts_model.load():
    # List available voices
    voices = tts_model.get_available_voices()
    
    # Set specific voice and parameters
    tts_model.set_voice_parameters(
        voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel (default)
        stability=0.7,
        similarity_boost=0.8
    )
```

#### Troubleshooting ElevenLabs

**API Key Issues:**
```bash
# Verify API key is set
echo $ELEVENLABS_API_KEY

# Test API connection
python -c "from elevenlabs import ElevenLabs; print('✅ Connected')"
```

**Network Issues:**
- Ensure internet connectivity
- Check firewall settings
- Verify ElevenLabs service status

**Usage Limits:**
- Monitor character usage in your ElevenLabs dashboard
- Free tier: 10,000 characters/month
- Paid plans available for higher usage

**Usage:**
1. Hold the assistant hotkey (Ctrl/Cmd + Mouse Back Button)
2. Speak your question or request
3. Release the hotkey
4. The AI response is copied to clipboard
5. If TTS is enabled, audio is generated and played automatically

### Platform-Specific Examples

**Linux/macOS:**
```bash
# Fast transcription only
holdtranscribe --model tiny --beam-size 1

# AI assistant with TTS
holdtranscribe --model mistralai/Voxtral-Mini-3B-2507 --tts
```

**Windows (Command Prompt):**
```cmd
# Fast transcription only
holdtranscribe --model tiny --beam-size 1

# AI assistant with TTS
holdtranscribe --model mistralai/Voxtral-Mini-3B-2507 --tts
```

**Windows (PowerShell):**
```powershell
# Fast transcription only
holdtranscribe --model tiny --beam-size 1

# AI assistant with TTS
holdtranscribe --model mistralai/Voxtral-Mini-3B-2507 --tts
```

---

## Auto-Start Service Setup

### Linux (systemd)

1. **Create service directory:**
   ```bash
   mkdir -p ~/.config/systemd/user
   ```

2. **Create service file:**
   ```bash
   cat > ~/.config/systemd/user/holdtranscribe.service << 'EOF'
   [Unit]
   Description=HoldTranscribe Voice Transcriber
   After=graphical-session.target

   [Service]
   Type=simple
   ExecStart=/usr/bin/holdtranscribe --model mistralai/Voxtral-Mini-3B-2507 --tts
   Restart=always
   RestartSec=5
   Environment=DISPLAY=:0
   Environment=XDG_RUNTIME_DIR=/run/user/%i
   WorkingDirectory=%h

   [Install]
   WantedBy=default.target
   EOF
   ```

3. **Enable and start:**
   ```bash
   systemctl --user daemon-reload
   systemctl --user enable holdtranscribe.service
   systemctl --user start holdtranscribe.service
   ```

### macOS (launchd)

1. **Create launch agent directory:**
   ```bash
   mkdir -p ~/Library/LaunchAgents
   ```

2. **Create plist file:**
   ```bash
   cat > ~/Library/LaunchAgents/com.holdtranscribe.plist << 'EOF'
   <?xml version="1.0" encoding="UTF-8"?>
   <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
   <plist version="1.0">
   <dict>
       <key>Label</key>
       <string>com.holdtranscribe</string>
       <key>ProgramArguments</key>
       <array>
           <string>/usr/local/bin/holdtranscribe</string>
           <string>--model</string>
           <string>mistralai/Voxtral-Mini-3B-2507</string>
           <string>--tts</string>
       </array>
       <key>RunAtLoad</key>
       <true/>
       <key>KeepAlive</key>
       <true/>
   </dict>
   </plist>
   EOF
   ```

3. **Load the service:**
   ```bash
   launchctl load ~/Library/LaunchAgents/com.holdtranscribe.plist
   launchctl start com.holdtranscribe
   ```

### Windows (Task Scheduler)

1. **Create batch file for easier management:**
   ```batch
   @echo off
   holdtranscribe --model mistralai/Voxtral-Mini-3B-2507 --tts
   ```
   Save as `holdtranscribe.bat`

2. **Using Task Scheduler GUI:**
   - Open Task Scheduler (taskschd.msc)
   - Create Basic Task → Name: "HoldTranscribe"
   - Trigger: When I log on
   - Action: Start a program → Browse to your batch file
   - Finish and test

3. **Using PowerShell (run as Administrator):**
   ```powershell
   $action = New-ScheduledTaskAction -Execute "C:\path\to\holdtranscribe.bat"
   $trigger = New-ScheduledTaskTrigger -AtLogon
   $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
   Register-ScheduledTask -TaskName "HoldTranscribe" -Action $action -Trigger $trigger -Settings $settings
   ```

---

## Configuration

### Hotkey Customization

Edit the `HOTKEY` set in the script to change key combinations:

```python
# Default: Ctrl + Mouse Forward Button
HOTKEY = {keyboard.Key.ctrl, mouse.Button.button9}

# Alternative examples:
# HOTKEY = {keyboard.Key.ctrl, keyboard.Key.space}  # Ctrl + Space
# HOTKEY = {keyboard.Key.alt, mouse.Button.left}    # Alt + Left Click
# HOTKEY = {mouse.Button.button8}                   # Mouse Back Button only
```

### Platform-Specific Mouse Button Notes

- **Windows:** Button numbers may vary by mouse driver
- **macOS:** Some mouse buttons may require additional permissions
- **Linux:** Button numbers can be checked with `xev` command

### Environment Variables

- `CUDA_VISIBLE_DEVICES` - Control GPU usage
- `TRANSFORMERS_CACHE` - Customize model cache location  
- `DISABLE_NOTIFY=1` - Suppress desktop notifications
- `PULSE_SERVER` (Linux) - Specify PulseAudio server
- `PORTAUDIO_DEVICE` - Force specific audio device

---

## Monitoring and Logs

### Linux (systemd)
```bash
# View logs
journalctl --user -u holdtranscribe.service -f

# Check status
systemctl --user status holdtranscribe.service
```

### macOS (launchd)
```bash
# View logs
tail -f ~/Library/Logs/com.holdtranscribe.log

# Check status
launchctl list | grep holdtranscribe
```

### Windows (Task Scheduler)
- Task Scheduler → Task Scheduler Library → HoldTranscribe → History tab
- Or check Windows Event Viewer → Applications and Services Logs

---

## Troubleshooting

### Common Issues (All Platforms)

**Model loading errors:**
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/transformers/
holdtranscribe --model tiny  # Start with smaller model
```

**Audio device issues:**
```bash
# List available devices
python -c "import sounddevice as sd; print(sd.query_devices())"
```

### Linux-Specific Issues

**Permission denied on input events:**
```bash
sudo usermod -aG input $USER
# Log out and back in
```

**Audio issues with PulseAudio:**
```bash
# Restart PulseAudio
pulseaudio -k
pulseaudio --start
```

**X11 forwarding issues:**
```bash
export DISPLAY=:0
xhost +local:
```

### macOS-Specific Issues

**Accessibility permissions denied:**
- System Preferences → Security & Privacy → Privacy → Accessibility
- Add Terminal or your Python executable
- May need to remove and re-add if issues persist

**Microphone access denied:**
- System Preferences → Security & Privacy → Privacy → Microphone
- Enable for Terminal/Python

**"Operation not permitted" errors:**
```bash
# Try running with sudo temporarily to identify permission issue
sudo holdtranscribe --debug
```

**Python/PortAudio conflicts:**
```bash
# Reinstall with Homebrew
brew uninstall portaudio
brew install portaudio
pip uninstall sounddevice
pip install sounddevice
```

### Windows-Specific Issues

**DLL load failures:**
```cmd
# Install Visual C++ Redistributable
# Download from Microsoft website
```

**Microphone access denied:**
- Settings → Privacy → Microphone → Allow apps to access microphone
- Ensure Python/Terminal is enabled

**CUDA issues:**
```cmd
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

**PowerShell execution policy:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Antivirus blocking:**
- Add Python executable to antivirus exclusions
- Add HoldTranscribe directory to exclusions

### Performance Optimization

**For slower systems:**
```bash
# Use fastest settings (transcription only)
holdtranscribe --model tiny --beam-size 1 --fast

# Lightweight AI assistant without TTS
holdtranscribe --model mistralai/Voxtral-Mini-3B-2507
```

**For better accuracy:**
```bash
# Use larger Whisper model
holdtranscribe --model large-v3 --beam-size 5

# Full AI assistant with TTS
holdtranscribe --model mistralai/Voxtral-Mini-3B-2507 --tts
```

**Memory management:**
```bash
# Monitor memory usage
holdtranscribe --debug

# Test TTS functionality
python examples/test_tts.py
```

### TTS-Specific Issues

**Dia installation problems:**
```bash
# Try installing transformers version first
pip install git+https://github.com/huggingface/transformers.git

# If that fails, try native Dia
pip install git+https://github.com/nari-labs/dia.git

# Test which implementation works
python examples/test_tts.py --method both
```

**Audio playback issues:**
```bash
# Linux: Install audio players
sudo apt install mpg123 # or vlc, mplayer

# macOS: Should work with default system player
# Windows: Should work with default system player

# Manual playback test
python examples/test_tts.py --text "Test speech generation"
```

**TTS performance optimization:**
```bash
# Use smaller TTS model (if available)
holdtranscribe --model mistralai/Voxtral-Mini-3B-2507 --tts --tts-model nari-labs/Dia-1.6B-0626

# Monitor TTS generation time
holdtranscribe --debug --tts
```

**TTS not working:**
```bash
# Check if Dia is properly installed
python -c "from dia.model import Dia; print('Native Dia: OK')"
python -c "from transformers import DiaForConditionalGeneration; print('Transformers Dia: OK')"

# Test with minimal example
python examples/test_tts.py --text "Hello world" --method native
```

---

## Contributing

Contributions, issues, and feature requests are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Test on multiple platforms when possible
4. Submit a pull request

When reporting issues, please include:
- Operating system and version
- Python version
- Full error message
- Steps to reproduce

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- OpenAI Whisper team for the excellent speech recognition model
- Contributors to the faster-whisper implementation
- All the open-source libraries that make this project possible