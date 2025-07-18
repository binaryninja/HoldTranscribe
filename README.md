# HoldTranscribe

Hotkey-Activated Voice-to-Clipboard Transcriber

A lightweight tool that records audio while you hold a configurable hotkey, transcribes speech using OpenAI's Whisper model, and copies the result to your clipboard.

---

## Features

* Hold-to-record using a customizable hotkey combination
* GPU acceleration with automatic CUDA detection and CPU fallback
* Instant copy of transcribed text to the clipboard
* Persistent model instance for low-latency transcription
* Configurable model size and beam search settings
* Detailed debug output and performance metrics
* Cross-platform support (Linux, macOS, Windows)
* Voice Activity Detection (VAD) for clean audio capture
* Auto-start service integration for all platforms

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
# Run with default settings
python voice_hold_to_clip.py

# Or if installed via pip
holdtranscribe
```

### Command Line Options

```bash
--model <size>       Whisper model size (tiny, base, small, medium, large-v3). Default: large-v3
--beam-size <n>      Beam search width (1 for fastest). Default: 5
--fast               Shorthand for `--model base --beam-size 1`
--debug              Enable verbose timing and resource metrics
--device <cpu|cuda>  Force CPU or GPU mode
```

### Platform-Specific Examples

**Linux/macOS:**
```bash
./voice_hold_to_clip.py --model tiny --beam-size 1
```

**Windows (Command Prompt):**
```cmd
python voice_hold_to_clip.py --model tiny --beam-size 1
```

**Windows (PowerShell):**
```powershell
python voice_hold_to_clip.py --model tiny --beam-size 1
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
   ExecStart=/usr/bin/python3 /path/to/holdtranscribe/voice_hold_to_clip.py --model large-v3 --beam-size 1
   Restart=always
   RestartSec=5
   Environment=DISPLAY=:0
   Environment=XDG_RUNTIME_DIR=/run/user/%i
   WorkingDirectory=/path/to/holdtranscribe

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
           <string>/usr/bin/python3</string>
           <string>/path/to/holdtranscribe/voice_hold_to_clip.py</string>
           <string>--model</string>
           <string>large-v3</string>
           <string>--beam-size</string>
           <string>1</string>
       </array>
       <key>RunAtLoad</key>
       <true/>
       <key>KeepAlive</key>
       <true/>
       <key>WorkingDirectory</key>
       <string>/path/to/holdtranscribe</string>
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
   cd /d "C:\path\to\holdtranscribe"
   python voice_hold_to_clip.py --model large-v3 --beam-size 1
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
python voice_hold_to_clip.py --model tiny  # Start with smaller model
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
sudo python voice_hold_to_clip.py --debug
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
# Use fastest settings
python voice_hold_to_clip.py --model tiny --beam-size 1 --fast
```

**For better accuracy:**
```bash
# Use larger model with more processing
python voice_hold_to_clip.py --model large-v3 --beam-size 5
```

**Memory management:**
```bash
# Monitor memory usage
python voice_hold_to_clip.py --debug
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