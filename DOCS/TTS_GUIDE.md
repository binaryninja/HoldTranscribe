# Text-to-Speech (TTS) Guide for HoldTranscribe

This guide covers the text-to-speech functionality in HoldTranscribe using the Dia model for generating natural-sounding speech from AI assistant responses.

## Overview

The TTS feature allows HoldTranscribe to:
- Generate speech audio from AI assistant responses
- Automatically play generated audio
- Save audio files for later use
- Support multiple Dia model implementations

## Installation

### Prerequisites

1. **Basic HoldTranscribe setup** (see main README.md)
2. **Voxtral AI assistant support** (for generating text responses)
3. **Dia TTS library** (for speech generation)

### Install Dia TTS Support

Choose one of these installation methods:

#### Option 1: Hugging Face Transformers (Recommended)
```bash
pip install git+https://github.com/huggingface/transformers.git
```

#### Option 2: Native Dia Implementation
```bash
pip install git+https://github.com/nari-labs/dia.git
```

#### Option 3: Both (Maximum Compatibility)
```bash
pip install git+https://github.com/huggingface/transformers.git
pip install git+https://github.com/nari-labs/dia.git
```

### Verify Installation

Test your TTS setup:
```bash
python examples/test_tts.py
```

## Basic Usage

### Enable TTS with AI Assistant

```bash
# Basic AI assistant with TTS
holdtranscribe --model mistralai/Voxtral-Mini-3B-2507 --tts

# With custom TTS model
holdtranscribe --model mistralai/Voxtral-Mini-3B-2507 --tts --tts-model nari-labs/Dia-1.6B-0626
```

### Hotkey Usage

1. **Hold the AI Assistant hotkey** (Ctrl/Cmd + Mouse Back Button)
2. **Speak your question** (e.g., "What's the weather like today?")
3. **Release the hotkey**
4. **Wait for processing**:
   - Text response appears in clipboard
   - Audio is generated (if TTS enabled)
   - Audio plays automatically
   - Desktop notification shows response

## Configuration Options

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--tts` | Enable TTS for assistant responses | Disabled |
| `--tts-model` | Dia model to use | `nari-labs/Dia-1.6B-0626` |
| `--tts-output` | Output filename template | `assistant_response.mp3` |

### Examples

```bash
# Enable TTS with default settings
holdtranscribe --model mistralai/Voxtral-Mini-3B-2507 --tts

# Custom TTS model
holdtranscribe --model mistralai/Voxtral-Mini-3B-2507 --tts --tts-model nari-labs/Dia-1.6B-0626

# Custom output file naming
holdtranscribe --model mistralai/Voxtral-Mini-3B-2507 --tts --tts-output "my_response.mp3"
```

## Text Formatting for TTS

The system automatically formats text for optimal speech generation:

### Automatic Formatting
- Plain text: `"Hello world"` → `"[S1] Hello world"`
- Already formatted: `"[S1] Hello world"` → No change

### Manual Speaker Tags
You can use multiple speakers in responses:
```
[S1] This is speaker one.
[S2] This is speaker two responding.
[S1] Speaker one again.
```

## Audio Output

### File Generation
- **Unique filenames**: Each response gets a unique ID to prevent conflicts
- **Format**: MP3 audio files
- **Location**: Current working directory (configurable)

### Automatic Playback
The system attempts to play generated audio using:
- **Linux**: `xdg-open`
- **macOS**: `open`
- **Windows**: `start`

### Manual Playback
If automatic playback fails, you can play files manually:
```bash
# Linux
mpg123 assistant_response_*.mp3
# or
vlc assistant_response_*.mp3

# macOS
open assistant_response_*.mp3

# Windows
start assistant_response_*.mp3
```

## Performance Considerations

### Generation Time
- **Typical**: 2-5 seconds for short responses
- **Longer text**: 5-10 seconds
- **GPU acceleration**: Significantly faster on CUDA-enabled systems

### Memory Usage
- **Dia model**: ~3-4 GB GPU memory
- **Combined with Voxtral**: 8-12 GB total
- **CPU fallback**: Higher RAM usage, slower generation

### Optimization Tips

```bash
# Monitor performance
holdtranscribe --debug --tts

# For slower systems - disable TTS
holdtranscribe --model mistralai/Voxtral-Mini-3B-2507

# For faster systems - enable TTS
holdtranscribe --model mistralai/Voxtral-Mini-3B-2507 --tts
```

## Troubleshooting

### TTS Not Working

#### Check Installation
```bash
# Test transformers implementation
python -c "from transformers import DiaForConditionalGeneration; print('Transformers: OK')"

# Test native implementation  
python -c "from dia.model import Dia; print('Native: OK')"
```

#### Test TTS Functionality
```bash
# Run comprehensive test
python examples/test_tts.py --method both

# Test specific implementation
python examples/test_tts.py --method transformers
python examples/test_tts.py --method native
```

#### Common Error Messages

**"Dia not available - skipping TTS model load"**
```bash
# Install Dia
pip install git+https://github.com/nari-labs/dia.git
```

**"Failed to load Dia with transformers"**
```bash
# Install/update transformers
pip install git+https://github.com/huggingface/transformers.git
```

**"TTS generation failed"**
```bash
# Check GPU memory
nvidia-smi

# Try CPU mode
holdtranscribe --device cpu --tts
```

### Audio Playback Issues

#### No Audio Output
1. **Check system audio settings**
2. **Verify audio file generation**:
   ```bash
   ls -la assistant_response_*.mp3
   ```
3. **Test manual playback**
4. **Check audio permissions**

#### Audio Quality Issues
- **Garbled output**: Check GPU memory availability
- **Robotic sound**: Normal for AI-generated speech
- **Clipping**: Adjust system volume levels

### Performance Issues

#### Slow TTS Generation
```bash
# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Monitor memory usage
holdtranscribe --debug --tts
```

#### Memory Issues
```bash
# Reduce memory usage
holdtranscribe --model mistralai/Voxtral-Mini-3B-2507 --device cpu --tts

# Check memory usage
holdtranscribe --debug
```

## Advanced Usage

### Custom TTS Integration

You can integrate TTS into custom scripts:

```python
from holdtranscribe.main import load_dia_model, generate_speech

# Load model
dia_model, dia_processor = load_dia_model("nari-labs/Dia-1.6B-0626", "cuda")

# Generate speech
success = generate_speech("Hello world!", "output.mp3")
```

### Batch TTS Generation

```bash
# Generate multiple responses
python examples/simple_tts.py "First response"
python examples/simple_tts.py "Second response"  
python examples/simple_tts.py "Third response"
```

### Service Integration

For always-on TTS service:

```bash
# Linux systemd
systemctl --user edit holdtranscribe.service

# Add TTS to service command
ExecStart=/usr/bin/holdtranscribe --model mistralai/Voxtral-Mini-3B-2507 --tts
```

## Best Practices

### Text Preparation
- **Keep responses conversational** for better speech quality
- **Use punctuation** to improve speech pacing
- **Avoid very long responses** (>500 words) for faster generation

### Resource Management
- **Monitor GPU memory** when using both Voxtral and Dia
- **Use CPU fallback** on memory-constrained systems
- **Clean up old audio files** periodically

### User Experience
- **Test TTS quality** with sample text before deployment
- **Configure notification settings** to match TTS timing
- **Set appropriate volume levels** for generated audio

## Examples

### Simple Question & Answer
```
User: "What's 2 plus 2?"
Assistant (text): "Two plus two equals four."
TTS Output: Natural speech saying "Two plus two equals four."
```

### Complex Conversation
```
User: "Explain quantum computing in simple terms"
Assistant (text): "Quantum computing uses quantum bits that can exist in multiple states simultaneously..."
TTS Output: Natural speech explanation with appropriate pacing
```

### Multi-turn Interaction
```
User: "What's the weather?"
Assistant: "I'd need to know your location to check the weather."
User: "I'm in New York"
Assistant: "I don't have access to real-time weather data, but you can check..."
```

## Support

### Getting Help
1. **Check debug output**: `holdtranscribe --debug --tts`
2. **Test TTS separately**: `python examples/test_tts.py`
3. **Review error messages** in console output
4. **Check system resources** (GPU memory, disk space)

### Reporting Issues
When reporting TTS-related issues, include:
- Operating system and version
- GPU model and memory
- Python and library versions
- Complete error messages
- TTS model being used
- Audio file generation success/failure

### Community Resources
- GitHub Issues: Report bugs and feature requests
- Discussions: Share usage patterns and tips
- Wiki: Community-maintained troubleshooting guides