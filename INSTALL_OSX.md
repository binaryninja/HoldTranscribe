# HoldTranscribe macOS Installation Guide

This guide provides step-by-step instructions for installing and configuring HoldTranscribe on macOS, including all the platform-specific requirements and troubleshooting steps.

## Prerequisites

### System Requirements
- **macOS**: 10.14 (Mojave) or later (macOS 11+ recommended)
- **Python**: 3.8 or later
- **Memory**: At least 4GB RAM (8GB+ recommended for larger models)
- **Storage**: 2-8GB free space (depending on model size)
- **Microphone**: Working microphone for voice input

### Command Line Tools
First, ensure you have the Xcode Command Line Tools installed:
```bash
xcode-select --install
```

## Step 1: Install Homebrew Dependencies

### Install Homebrew (if not already installed)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Install Required System Libraries
```bash
# Install D-Bus (required for notifications)
brew install dbus

# Install PortAudio (required for audio processing)
brew install portaudio

# Install pkg-config (required for building Python extensions)
brew install pkg-config
```

### Set up D-Bus Service
```bash
# Copy the D-Bus launch agent (optional - for system notifications)
sudo cp /opt/homebrew/Cellar/dbus/*/lib/Library/LaunchAgents/org.freedesktop.dbus-session.plist /Library/LaunchAgents
sudo chmod 644 /Library/LaunchAgents/org.freedesktop.dbus-session.plist
```

## Step 2: Install Python Dependencies

### Set Environment Variables
```bash
# Set PKG_CONFIG_PATH to help find libraries
export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:/opt/homebrew/share/pkgconfig:$PKG_CONFIG_PATH"

# Add to your shell profile for persistence
echo 'export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:/opt/homebrew/share/pkgconfig:$PKG_CONFIG_PATH"' >> ~/.zshrc
```

### Install D-Bus Python Bindings
```bash
pip install dbus-python
```

### Install HoldTranscribe
Choose one of these methods:

**Option 1: From PyPI (Recommended)**
```bash
pip install holdtranscribe
```

**Option 2: From GitHub**
```bash
pip install git+https://github.com/binaryninja/holdtranscribe.git
```

**Option 3: Development Installation**
```bash
git clone https://github.com/binaryninja/holdtranscribe.git
cd holdtranscribe
pip install -e .
```

## Step 3: Configure macOS Permissions

HoldTranscribe requires several system permissions to function properly:

### 1. Accessibility Access
1. Open **System Settings** (or **System Preferences** on older macOS)
2. Navigate to **Privacy & Security**
3. Click **Accessibility** in the left sidebar
4. Click the **+** button and add **Terminal** (or your Python executable)
5. Ensure the checkbox next to Terminal is **enabled**

### 2. Input Monitoring
1. In **System Settings → Privacy & Security**
2. Click **Input Monitoring** in the left sidebar
3. Click the **+** button and add **Terminal**
4. Ensure the checkbox next to Terminal is **enabled**

### 3. Microphone Access
This permission is usually requested automatically when you first try to record audio, but you can also set it manually:
1. In **System Settings → Privacy & Security**
2. Click **Microphone** in the left sidebar
3. Ensure **Terminal** is enabled (it may appear automatically after first use)

### Important Notes:
- You may need to **restart Terminal** after granting permissions
- Some applications may require you to **remove and re-add** the Terminal app if permissions aren't working
- The exact location of these settings may vary depending on your macOS version

## Step 4: Configure Hotkeys for macOS

Due to macOS limitations, the default mouse button hotkey (`mouse.Button.button9`) doesn't work. You'll need to modify the hotkey configuration.

### Quick Fix: Use F9 Key
If you installed via development mode (Option 3 above), you can modify the hotkey:

1. Edit `holdtranscribe/main.py`
2. Find the line: `HOTKEY = {keyboard.Key.ctrl, mouse.Button.button9}`
3. Change it to: `HOTKEY = {keyboard.Key.f9}`
4. Also update the message from: `"Hold Ctrl+Mouse Forward Button to speak"`
5. To: `"Hold F9 to speak"`

### Alternative Hotkey Options
You can use any of these combinations:
```python
# Single keys
HOTKEY = {keyboard.Key.f9}                    # F9 key
HOTKEY = {keyboard.Key.f10}                   # F10 key

# Key combinations
HOTKEY = {keyboard.Key.ctrl, keyboard.Key.space}    # Ctrl + Space
HOTKEY = {keyboard.Key.alt, keyboard.Key.space}     # Alt + Space
HOTKEY = {keyboard.Key.cmd, keyboard.Key.space}     # Cmd + Space

# Mouse buttons (basic ones work)
HOTKEY = {keyboard.Key.ctrl, mouse.Button.right}    # Ctrl + Right Click
HOTKEY = {keyboard.Key.alt, mouse.Button.left}      # Alt + Left Click
```

## Step 5: Test the Installation

### Basic Test
```bash
# Test that the command is available
holdtranscribe --help
```

### Audio Device Test
```bash
# Check available audio devices
python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

### Full Test with Debug Mode
```bash
# Run with smallest model and debug output
holdtranscribe --model tiny --debug
```

You should see:
- No import errors
- Model loading successfully
- "Hold [your hotkey] to speak" message
- No "This process is not trusted!" errors

## Step 6: Usage

1. **Start the application**: `holdtranscribe --model tiny`
2. **Hold your hotkey**: Press and hold your configured hotkey (e.g., F9)
3. **Speak**: Talk while holding the hotkey
4. **Release**: Let go of the hotkey to stop recording
5. **Result**: Transcribed text is automatically copied to your clipboard

## Troubleshooting

### Common Issues

#### "This process is not trusted!" Error
- **Solution**: Grant Accessibility and Input Monitoring permissions (Step 3)
- **Additional step**: Restart Terminal after granting permissions

#### "No module named 'dbus'" Error
- **Solution**: Install D-Bus and dbus-python:
```bash
brew install dbus
export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:/opt/homebrew/share/pkgconfig:$PKG_CONFIG_PATH"
pip install dbus-python
```

#### "AttributeError: type object 'Button' has no attribute 'button9'"
- **Solution**: Modify hotkey configuration to use macOS-compatible keys (Step 4)

#### Audio Device Errors
```bash
# Check audio devices
python3 -c "import sounddevice as sd; print(sd.query_devices())"

# If no devices found, check system audio settings
```

#### Microphone Permission Denied
- **Solution**: 
  1. Check Privacy & Security → Microphone settings
  2. Try running once to trigger permission request
  3. Restart Terminal if needed

#### Model Loading Errors
```bash
# Clear cache and try smaller model
rm -rf ~/.cache/huggingface/transformers/
holdtranscribe --model tiny
```

#### D-Bus Notification Errors
This is normal on macOS. Notifications may not work perfectly, but the core functionality (transcription and clipboard) will work fine.

### Performance Issues

#### For slower Macs:
```bash
# Use fastest settings
holdtranscribe --model tiny --beam-size 1 --fast
```

#### For better accuracy:
```bash
# Use larger model (requires more memory)
holdtranscribe --model large-v3 --beam-size 5
```

#### Memory monitoring:
```bash
# Monitor memory usage
holdtranscribe --debug --model tiny
```

## Step 7: Auto-Start Setup (Optional)

### Create Launch Agent
```bash
# Create directory
mkdir -p ~/Library/LaunchAgents

# Create plist file (adjust paths as needed)
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
        <string>-m</string>
        <string>holdtranscribe</string>
        <string>--model</string>
        <string>tiny</string>
        <string>--fast</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
EOF
```

### Load the Service
```bash
# Load the launch agent
launchctl load ~/Library/LaunchAgents/com.holdtranscribe.plist

# Start the service
launchctl start com.holdtranscribe
```

### Manage the Service
```bash
# Check status
launchctl list | grep holdtranscribe

# Stop service
launchctl stop com.holdtranscribe

# Unload service
launchctl unload ~/Library/LaunchAgents/com.holdtranscribe.plist
```

## Model Selection Guide

| Model | Size | Speed | Accuracy | Memory | Best For |
|-------|------|-------|----------|---------|----------|
| tiny | ~39MB | Fastest | Basic | ~1GB | Testing, older Macs |
| base | ~74MB | Fast | Good | ~1GB | Daily use, balanced |
| small | ~244MB | Medium | Better | ~2GB | Better accuracy |
| medium | ~769MB | Slower | Good | ~5GB | High accuracy |
| large-v3 | ~1550MB | Slowest | Best | ~10GB | Maximum accuracy |

## Environment Variables

You can customize behavior with these environment variables:

```bash
# Use specific GPU (if available)
export CUDA_VISIBLE_DEVICES=0

# Custom model cache location
export TRANSFORMERS_CACHE=/path/to/cache

# Disable desktop notifications
export DISABLE_NOTIFY=1

# Force specific audio device
export PORTAUDIO_DEVICE=1
```

## Uninstallation

To remove HoldTranscribe:

```bash
# Remove the package
pip uninstall holdtranscribe

# Remove downloaded models (optional)
rm -rf ~/.cache/huggingface/transformers/

# Remove launch agent (if configured)
launchctl unload ~/Library/LaunchAgents/com.holdtranscribe.plist
rm ~/Library/LaunchAgents/com.holdtranscribe.plist
```

## Getting Help

If you encounter issues:

1. **Run with debug mode**: `holdtranscribe --debug --model tiny`
2. **Check the logs**: Look for error messages in the terminal output
3. **Verify permissions**: Ensure all three permissions are granted
4. **Test audio**: Verify your microphone works with other applications
5. **GitHub Issues**: https://github.com/binaryninja/holdtranscribe/issues

## What's Next?

After successful installation:
1. **Experiment with models**: Try different model sizes for speed vs. accuracy
2. **Customize hotkeys**: Find a key combination that works best for you
3. **Set up auto-start**: Configure it to start automatically on login
4. **Integrate with workflows**: Use with your favorite text editor or note-taking app

Enjoy using HoldTranscribe on macOS! 🎤✨