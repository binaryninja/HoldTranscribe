# HoldTranscribe Installation Guide

This guide will help you install HoldTranscribe, a hotkey-activated voice-to-clipboard transcriber, using pip.

## Prerequisites

### System Requirements

- **Python**: 3.8 or later
- **Operating System**: Linux, macOS, or Windows
- **Audio System**: Working microphone and audio system
- **Memory**: At least 4GB RAM (8GB+ recommended for larger models)
- **Storage**: 2-8GB free space (depending on model size)

### Platform-Specific Prerequisites

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3-pip portaudio19-dev pulseaudio build-essential
```

#### Linux (Fedora/RHEL/CentOS)
```bash
sudo dnf install python3-pip portaudio-devel pulseaudio gcc gcc-c++
```

#### macOS
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install PortAudio
brew install portaudio

# For Apple Silicon Macs, consider using conda
# conda install python=3.9
```

#### Windows
- Install Python from [python.org](https://python.org) or Microsoft Store
- Ensure "Add Python to PATH" is checked during installation
- Install Microsoft C++ Build Tools if compilation errors occur

## Installation Methods

### Method 1: Install from PyPI (Recommended)

```bash
pip install holdtranscribe
```

### Method 2: Install from GitHub

```bash
# Latest stable release
pip install git+https://github.com/binaryninja/holdtranscribe.git

# Specific version (replace v1.0.0 with desired version)
pip install git+https://github.com/binaryninja/holdtranscribe.git@v1.0.0
```

### Method 3: Development Installation

```bash
# Clone the repository
git clone https://github.com/binaryninja/holdtranscribe.git
cd holdtranscribe

# Install in editable mode
pip install -e .
```

## Optional GPU Acceleration

For NVIDIA GPU acceleration (significant performance boost):

```bash
# Install GPU support
pip install holdtranscribe[gpu]

# Or manually install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Verification

Test that the installation was successful:

```bash
# Check if the command is available
holdtranscribe --help

# Test with debug mode (safe to interrupt with Ctrl+C)
holdtranscribe --debug --model tiny
```

You should see:
- No import errors
- Model loading successfully
- "Hold Ctrl+Mouse Forward Button to speak" message

## Platform-Specific Setup

### Linux Setup

1. **Add user to input group** (if permission errors occur):
   ```bash
   sudo usermod -aG input $USER
   # Log out and back in for changes to take effect
   ```

2. **Test audio devices**:
   ```bash
   python3 -c "import sounddevice as sd; print(sd.query_devices())"
   ```

### macOS Setup

1. **Grant permissions** in System Preferences:
   - **Microphone Access**: Security & Privacy → Privacy → Microphone
   - **Accessibility Access**: Security & Privacy → Privacy → Accessibility
   - **Input Monitoring**: Security & Privacy → Privacy → Input Monitoring

2. **Test permissions**:
   ```bash
   # This should show available audio devices without errors
   python3 -c "import sounddevice as sd; print(sd.query_devices())"
   ```

### Windows Setup

1. **Grant microphone permissions**:
   - Settings → Privacy → Microphone → Allow apps to access microphone

2. **Test installation**:
   ```cmd
   holdtranscribe --help
   ```

## Quick Start

1. **Basic usage** (default settings):
   ```bash
   holdtranscribe
   ```

2. **Fast mode** (lower accuracy, faster processing):
   ```bash
   holdtranscribe --fast
   ```

3. **Custom model** (balance speed vs accuracy):
   ```bash
   # Fastest
   holdtranscribe --model tiny --beam-size 1
   
   # Most accurate
   holdtranscribe --model large-v3 --beam-size 5
   ```

4. **Debug mode** (troubleshooting):
   ```bash
   holdtranscribe --debug
   ```

## Usage Instructions

1. **Start the application**: Run `holdtranscribe` in your terminal
2. **Hold hotkey**: Press and hold `Ctrl + Mouse Forward Button` (or `Ctrl + Mouse Button 9`)
3. **Speak**: Talk while holding the hotkey
4. **Release**: Let go of the hotkey to stop recording
5. **Result**: Transcribed text is automatically copied to your clipboard

### Default Hotkey

- **Default**: `Ctrl + Mouse Forward Button` (button 9)
- **Alternative**: You can modify the source code to change hotkeys

### Available Models

| Model | Size | Speed | Accuracy | Memory |
|-------|------|-------|----------|---------|
| tiny | ~39MB | Fastest | Lowest | ~1GB |
| base | ~74MB | Fast | Good | ~1GB |
| small | ~244MB | Medium | Better | ~2GB |
| medium | ~769MB | Slower | Good | ~5GB |
| large-v3 | ~1550MB | Slowest | Best | ~10GB |

## Troubleshooting

### Common Issues

#### "No module named 'holdtranscribe'"
```bash
# Ensure you're in the correct environment
pip list | grep holdtranscribe

# Reinstall if missing
pip install --upgrade holdtranscribe
```

#### Audio device errors
```bash
# List available audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Linux: Restart PulseAudio
pulseaudio -k && pulseaudio --start

# macOS: Check microphone permissions in System Preferences
```

#### Model loading errors
```bash
# Clear cache and try smaller model
rm -rf ~/.cache/huggingface/transformers/
holdtranscribe --model tiny
```

#### Permission denied (Linux)
```bash
# Add user to input group
sudo usermod -aG input $USER
# Log out and back in
```

#### GPU not detected
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install GPU support
pip install holdtranscribe[gpu]
```

### Performance Optimization

#### For slower systems:
```bash
holdtranscribe --model tiny --beam-size 1
```

#### For better accuracy:
```bash
holdtranscribe --model large-v3 --beam-size 5
```

#### Memory issues:
```bash
# Use smaller model
holdtranscribe --model base

# Monitor memory usage
holdtranscribe --debug
```

## Uninstallation

To remove HoldTranscribe:

```bash
pip uninstall holdtranscribe
```

To also remove downloaded models:
```bash
# Linux/macOS
rm -rf ~/.cache/huggingface/transformers/

# Windows
rmdir /s %USERPROFILE%\.cache\huggingface\transformers\
```

## Advanced Configuration

### Environment Variables

- `CUDA_VISIBLE_DEVICES=0` - Use specific GPU
- `TRANSFORMERS_CACHE=/path/to/cache` - Custom model cache location
- `DISABLE_NOTIFY=1` - Suppress desktop notifications

### Custom Hotkeys

To change hotkeys, you'll need to modify the source code:

```python
# In holdtranscribe/main.py
HOTKEY = {keyboard.Key.ctrl, keyboard.Key.space}  # Ctrl + Space
HOTKEY = {keyboard.Key.alt, mouse.Button.left}    # Alt + Left Click
HOTKEY = {mouse.Button.button8}                   # Mouse Back Button only
```

## Getting Help

- **GitHub Issues**: https://github.com/binaryninja/holdtranscribe/issues
- **Documentation**: https://github.com/binaryninja/holdtranscribe#readme
- **Debug mode**: Run with `--debug` flag for detailed logging

## What's Next?

After installation, you might want to:

1. **Auto-start on boot**: Set up system service (see main README)
2. **Customize hotkeys**: Modify source for different key combinations
3. **Optimize performance**: Experiment with different model sizes
4. **Integration**: Use with your favorite text editor or note-taking app

Enjoy using HoldTranscribe! 🎤✨