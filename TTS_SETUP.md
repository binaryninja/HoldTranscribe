# TTS Setup Guide for HoldTranscribe

This guide explains how to set up and use the Text-to-Speech (TTS) functionality in HoldTranscribe, which now uses ElevenLabs as the default TTS implementation.

## Overview

The TTS implementation provides high-quality text-to-speech synthesis with multiple model options:

- **ElevenLabs TTS** (Default) - Cloud-based, high-quality synthesis with various voices
- **Moshi TTS** - Local, real-time streaming synthesis using Kyutai models
- **DIA TTS** - Local synthesis using Facebook's models

## ElevenLabs TTS Setup (Default)

### Requirements

- **ElevenLabs API Key** - Sign up at [elevenlabs.io](https://elevenlabs.io)
- **Internet Connection** - Required for cloud-based synthesis
- **Python**: 3.8 or higher
- **RAM**: 2GB+ (minimal, as processing is cloud-based)

### Installation

#### 1. Install ElevenLabs Package

```bash
# Navigate to HoldTranscribe directory
cd HoldTranscribe

# Install TTS dependencies (includes ElevenLabs)
pip install -r requirements-tts.txt

# Or install ElevenLabs package directly
pip install elevenlabs>=1.0.0
```

#### 2. Get Your API Key

1. Visit [ElevenLabs](https://elevenlabs.io/app/settings/api-keys)
2. Sign up for an account (free tier available)
3. Generate an API key from your settings

#### 3. Set Your API Key

```bash
# Set as environment variable (recommended)
export ELEVENLABS_API_KEY="your_api_key_here"

# Or add to your shell profile (.bashrc, .zshrc, etc.)
echo 'export ELEVENLABS_API_KEY="your_api_key_here"' >> ~/.bashrc
source ~/.bashrc
```

#### 4. Verify Installation

```bash
# Run the ElevenLabs integration test
python test_elevenlabs_integration.py

# Run the example script
python examples/elevenlabs_tts_example.py
```

### Usage

#### Command Line Interface

```bash
# Basic usage with ElevenLabs TTS (default)
python -m holdtranscribe.main --tts

# Specify ElevenLabs model explicitly
python -m holdtranscribe.main --tts --tts-model="eleven_multilingual_v2"

# Use different ElevenLabs models
python -m holdtranscribe.main --tts --tts-model="eleven_turbo_v2_5"  # Faster
python -m holdtranscribe.main --tts --tts-model="eleven_flash_v2_5"  # Fastest
```

#### Python API

```python
from holdtranscribe.models import ModelFactory
import os

# Create ElevenLabs TTS model (default)
tts_model = ModelFactory.create_tts_model(
    "default",  # Uses ElevenLabs eleven_multilingual_v2
    device="cpu",  # Device parameter (not used for cloud API)
    api_key=os.getenv("ELEVENLABS_API_KEY")
)

if tts_model and tts_model.load():
    # Generate audio file
    success = tts_model.synthesize("Hello, world!", "output.mp3")
    
    # Streaming synthesis (for real-time applications)
    audio_stream = tts_model.synthesize_streaming("Long text here...")
    if audio_stream:
        for chunk in audio_stream:
            # Process audio chunk in real-time
            pass
    
    # Clean up
    tts_model.unload()
```

### Configuration

#### Voice Selection

```python
# Get available voices
voices = tts_model.get_available_voices()
for voice in voices:
    print(f"Voice: {voice['name']} (ID: {voice['voice_id']})")

# Set a specific voice
tts_model.set_voice_parameters(
    voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel (default)
    stability=0.7,          # Voice stability (0.0-1.0)
    similarity_boost=0.8,   # Voice similarity (0.0-1.0)
    style=0.0,             # Style exaggeration (0.0-1.0)
    use_speaker_boost=True  # Speaker boost enhancement
)
```

#### Streaming Parameters

```python
# Configure streaming for lower latency
tts_model.set_streaming_parameters(
    chunk_size=1024,                    # Audio chunk size
    optimize_streaming_latency=3        # Latency optimization (0-4)
)
```

#### Model Selection

```python
# Available ElevenLabs models
models = {
    "eleven_multilingual_v2": "High quality, supports multiple languages",
    "eleven_turbo_v2_5": "Faster synthesis, good quality",
    "eleven_flash_v2_5": "Fastest synthesis, lower latency",
    "eleven_multilingual_v1": "Legacy multilingual model"
}

# Create specific model
tts_model = ModelFactory.create_tts_model(
    "eleven_turbo_v2_5",  # Fast model
    device="cpu",
    api_key=os.getenv("ELEVENLABS_API_KEY")
)
```

### Integration with HoldTranscribe

#### Basic Integration

```bash
# Enable TTS in HoldTranscribe (uses ElevenLabs by default)
python -m holdtranscribe.main --tts

# The assistant responses will now be spoken aloud using ElevenLabs
```

#### Advanced Configuration

```python
from holdtranscribe.app import HoldTranscribeApp
import argparse

# Configure app with custom TTS settings
app = HoldTranscribeApp()
args = argparse.Namespace()
args.tts = True
args.tts_model = "eleven_turbo_v2_5"  # Use faster model
args.model = "mistralai/Voxtral-Mini-3B-2507"  # Assistant model

app.args = args
if app.initialize_models():
    # TTS is now ready with ElevenLabs
    app.run()
```

## Alternative TTS Models

### Moshi TTS (Local)

For offline usage or when you prefer local processing:

```bash
# Use Moshi TTS instead of ElevenLabs
python -m holdtranscribe.main --tts --tts-model="kyutai/moshiko-pytorch-bf16"

# Requires additional setup - see Moshi section below
pip install moshi>=0.2.10 sphn>=0.1.0
```

### DIA TTS (Local)

For Facebook's DIA models:

```bash
# Use DIA TTS
python -m holdtranscribe.main --tts --tts-model="facebook/mms-tts-eng"
```

## Troubleshooting

### Common ElevenLabs Issues

#### 1. API Key Problems

```bash
# Check if API key is set
echo $ELEVENLABS_API_KEY

# Test API key validity
python -c "
import os
from elevenlabs import ElevenLabs
client = ElevenLabs(api_key=os.getenv('ELEVENLABS_API_KEY'))
print('API key is valid:', bool(client.user.get()))
"
```

#### 2. Network/Connection Issues

```bash
# Test internet connectivity
ping api.elevenlabs.io

# Check firewall/proxy settings
curl -H "Accept: application/json" https://api.elevenlabs.io/v1/user

# Test with verbose output
python examples/elevenlabs_tts_example.py
```

#### 3. Audio Output Issues

```bash
# Check if audio file was generated
ls -la assistant_response_*.mp3

# Test audio playback
# On Linux:
aplay assistant_response_*.mp3  # or
paplay assistant_response_*.mp3

# On macOS:
afplay assistant_response_*.mp3

# On Windows:
# Use any media player or:
# powershell -c "(New-Object Media.SoundPlayer 'assistant_response.mp3').PlaySync()"
```

#### 4. Quota/Billing Issues

```bash
# Check your usage and limits
python -c "
import os
from elevenlabs import ElevenLabs
client = ElevenLabs(api_key=os.getenv('ELEVENLABS_API_KEY'))
user = client.user.get()
print(f'Characters used: {user.subscription.character_count}')
print(f'Character limit: {user.subscription.character_limit}')
"
```

### Error Messages

#### "ElevenLabs package not installed"
```bash
pip install elevenlabs>=1.0.0
```

#### "ElevenLabs API key is required"
```bash
export ELEVENLABS_API_KEY="your_api_key_here"
```

#### "Failed to connect to ElevenLabs API"
- Check internet connection
- Verify API key is correct
- Check if ElevenLabs service is available

#### "Model not found" or "Voice not found"
```python
# List available models and voices
from holdtranscribe.models import ModelFactory
model = ModelFactory.create_tts_model("elevenlabs", "cpu", api_key="your_key")
if model.load():
    voices = model.get_available_voices()
    print("Available voices:", [v['name'] for v in voices])
```

## Moshi TTS Setup (Alternative)

If you prefer local TTS processing:

### Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 8GB+ (16GB+ recommended for GPU)
- **Storage**: 5GB+ for model downloads

### Installation
```bash
# Install Moshi dependencies
pip install moshi>=0.2.10 sphn>=0.1.0 torch>=2.0.0

# Test Moshi TTS
python test_moshi_tts.py
```

### Usage
```bash
# Use Moshi instead of ElevenLabs
python -m holdtranscribe.main --tts --tts-model="moshi"
```

## Performance Comparison

| TTS Model | Speed | Quality | Requirements | Offline |
|-----------|-------|---------|--------------|---------|
| ElevenLabs | Very Fast | Excellent | API Key, Internet | No |
| Moshi | Fast | Very Good | GPU Recommended | Yes |
| DIA | Medium | Good | CPU/GPU | Yes |

## API Reference

### ElevenLabsTTSWrapper Methods

- `load()` - Initialize API connection
- `unload()` - Clean up resources
- `synthesize(text, output_file)` - Generate audio file
- `synthesize_streaming(text)` - Stream audio chunks
- `get_available_voices()` - List available voices
- `set_voice_parameters(voice_id, **settings)` - Configure voice
- `set_streaming_parameters(**params)` - Configure streaming
- `get_model_info()` - Get model information

### Environment Variables

- `ELEVENLABS_API_KEY` - Your ElevenLabs API key (required)

### Configuration Options

```python
# Model creation options
model = ModelFactory.create_tts_model(
    model_name="eleven_multilingual_v2",
    device="cpu",  # Not used for ElevenLabs
    api_key="your_api_key",
    voice_id="21m00Tcm4TlvDq8ikWAM",  # Default voice
    voice_settings={
        "stability": 0.5,
        "similarity_boost": 0.5,
        "style": 0.0,
        "use_speaker_boost": True
    },
    output_format="mp3_44100_128"  # Audio format
)
```

## Support

For issues and questions:

1. Check this troubleshooting guide
2. Run the test script: `python test_elevenlabs_integration.py`
3. Verify your API key and internet connection
4. Check ElevenLabs API status
5. Report issues on the HoldTranscribe repository

## Pricing and Usage

ElevenLabs offers:
- **Free Tier**: 10,000 characters/month
- **Starter**: $5/month for 30,000 characters
- **Creator**: $22/month for 100,000 characters
- **Pro**: $99/month for 500,000 characters

See [ElevenLabs Pricing](https://elevenlabs.io/pricing) for current rates.

## Best Practices

1. **Cache API Key**: Set as environment variable, don't hardcode
2. **Monitor Usage**: Track character consumption to avoid overage
3. **Optimize Text**: Break long texts into chunks for better streaming
4. **Error Handling**: Always check synthesis success before playing audio
5. **Voice Selection**: Test different voices to find the best fit for your use case
6. **Latency Optimization**: Use faster models (turbo/flash) for real-time applications

## Advanced Features

### Custom Voice Settings per Text

```python
# Adjust voice for different types of content
if text.endswith('?'):
    # More expressive for questions
    voice_settings = {"stability": 0.3, "similarity_boost": 0.7, "style": 0.2}
else:
    # Stable for statements
    voice_settings = {"stability": 0.7, "similarity_boost": 0.8, "style": 0.0}

tts_model.set_voice_parameters(voice_id="your_voice", **voice_settings)
```

### Batch Processing

```python
texts = ["First sentence.", "Second sentence.", "Third sentence."]

for i, text in enumerate(texts):
    output_file = f"output_{i:03d}.mp3"
    success = tts_model.synthesize(text, output_file)
    if success:
        print(f"Generated: {output_file}")
```

### Real-time Streaming Integration

```python
import queue
import threading

def real_time_tts(text_queue, audio_callback):
    while True:
        text = text_queue.get()
        if text is None:  # Stop signal
            break
            
        audio_stream = tts_model.synthesize_streaming(text)
        if audio_stream:
            for chunk in audio_stream:
                audio_callback(chunk)
```

This completes the setup guide for using ElevenLabs TTS as the default text-to-speech engine in HoldTranscribe.