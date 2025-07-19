# ElevenLabs TTS Integration for HoldTranscribe

This document describes how to use ElevenLabs Text-to-Speech (TTS) with HoldTranscribe for high-quality voice synthesis.

## Overview

ElevenLabs provides state-of-the-art text-to-speech synthesis with:
- Natural-sounding voices
- Multiple language support
- Voice cloning capabilities
- Real-time streaming synthesis
- Fine-grained voice control

## Installation

### Prerequisites

1. **ElevenLabs Account**: Sign up at [ElevenLabs](https://elevenlabs.io)
2. **API Key**: Get your API key from the [ElevenLabs dashboard](https://elevenlabs.io/app/settings/api-keys)

### Install Dependencies

```bash
# Install ElevenLabs package
pip install elevenlabs>=1.0.0

# Or install from requirements
pip install -r requirements.txt
```

## Configuration

### API Key Setup

Set your ElevenLabs API key as an environment variable:

```bash
# Linux/macOS
export ELEVENLABS_API_KEY="your_api_key_here"

# Windows (Command Prompt)
set ELEVENLABS_API_KEY=your_api_key_here

# Windows (PowerShell)
$env:ELEVENLABS_API_KEY="your_api_key_here"
```

Or create a `.env` file in your project root:

```env
ELEVENLABS_API_KEY=your_api_key_here
```

## Usage

### Basic Usage

```python
from holdtranscribe.models import ModelFactory, ModelType

# Create ElevenLabs TTS model
model = ModelFactory.create_tts_model(
    model_name="eleven_multilingual_v2",
    device="cpu",  # Not used for cloud API
    api_key="your_api_key"  # Optional if set in environment
)

# Load the model
if model.load():
    # Synthesize text to speech
    success = model.synthesize(
        text="Hello, this is ElevenLabs speaking!",
        output_file="output.mp3"
    )
    
    if success:
        print("Audio generated successfully!")
    
    # Unload when done
    model.unload()
```

### Using Different Voices

```python
# Get available voices
voices = model.get_available_voices()

for voice in voices[:5]:  # Show first 5 voices
    print(f"Voice: {voice['name']} (ID: {voice['voice_id']})")

# Set a specific voice
model.set_voice_parameters(
    voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel voice
    stability=0.7,
    similarity_boost=0.8,
    style=0.2
)
```

### Streaming Synthesis

```python
# Configure for streaming
model.set_streaming_parameters(
    optimize_streaming_latency=3,  # Max optimization (0-4)
    chunk_size=1024
)

# Stream audio synthesis
audio_stream = model.synthesize_streaming(
    text="This text will be synthesized in real-time chunks."
)

if audio_stream:
    with open("streaming_output.mp3", "wb") as f:
        for chunk in audio_stream:
            if chunk:
                f.write(chunk)
```

## Model Names and Selection

You can use ElevenLabs TTS with various model names:

### Supported Model Names

- `eleven_multilingual_v2` - High-quality multilingual model
- `eleven_turbo_v2_5` - Fast model optimized for streaming
- `eleven_flash_v2_5` - Ultra-fast model for real-time applications
- `elevenlabs` - Default ElevenLabs model
- `eleven_*` - Any model starting with "eleven_"

### Model Selection Examples

```python
# High-quality model
model = ModelFactory.create_tts_model("eleven_multilingual_v2", "cpu")

# Fast streaming model
model = ModelFactory.create_tts_model("eleven_turbo_v2_5", "cpu")

# Generic ElevenLabs (uses default model)
model = ModelFactory.create_tts_model("elevenlabs", "cpu")
```

## Voice Settings

### Voice Parameters

```python
model.set_voice_parameters(
    voice_id="voice_id_here",
    stability=0.5,        # 0.0-1.0, higher = more stable
    similarity_boost=0.5, # 0.0-1.0, higher = more similar to original
    style=0.0,            # 0.0-1.0, style exaggeration
    use_speaker_boost=True # Enable speaker boost
)
```

### Output Formats

Supported output formats:
- `mp3_44100_128` (default)
- `mp3_44100_192`
- `pcm_16000`
- `pcm_22050`
- `pcm_44100`
- `ulaw_8000`

```python
# Set output format
success = model.synthesize(
    text="Hello world!",
    output_file="output.wav",
    output_format="pcm_44100"
)
```

## Advanced Features

### Voice Consistency

For multi-chunk synthesis with consistent voice characteristics:

```python
# Enable voice consistency
model.set_voice_conditioning(enabled=True)

# Synthesize with previous context
model.synthesize(
    text="This is the second sentence.",
    output_file="part2.mp3",
    previous_text="This was the first sentence."
)
```

### Streaming with Latency Optimization

```python
# Configure for real-time streaming
model.set_streaming_parameters(
    optimize_streaming_latency=4,  # Maximum optimization
    chunk_size=512                  # Smaller chunks for lower latency
)

# Stream with optimizations
for chunk in model.synthesize_streaming(text, optimize_streaming_latency=4):
    # Process audio chunk immediately
    play_audio_chunk(chunk)
```

### Custom Voice Settings per Request

```python
from elevenlabs import VoiceSettings

custom_settings = VoiceSettings(
    stability=0.8,
    similarity_boost=0.9,
    style=0.1,
    use_speaker_boost=True
)

model.synthesize(
    text="Custom voice settings example",
    output_file="custom.mp3",
    voice_settings=custom_settings
)
```

## API Reference

### ModelFactory.create_tts_model()

Creates an ElevenLabs TTS model instance.

**Parameters:**
- `model_name` (str): ElevenLabs model name
- `device` (str): Device parameter (ignored for cloud API)
- `api_key` (str, optional): ElevenLabs API key
- `voice_id` (str, optional): Default voice ID
- `voice_settings` (dict, optional): Default voice settings
- `output_format` (str, optional): Default output format

### ElevenLabsTTSWrapper Methods

#### `load() -> bool`
Initialize connection to ElevenLabs API.

#### `unload()`
Clean up resources.

#### `synthesize(text: str, output_file: str, **kwargs) -> bool`
Synthesize text to speech and save to file.

**Parameters:**
- `text`: Text to synthesize
- `output_file`: Output file path
- `voice_id`: Override default voice
- `voice_settings`: Override voice settings
- `output_format`: Override output format
- `seed`: Random seed for reproducible results
- `previous_text`: Previous text for voice consistency

#### `synthesize_streaming(text: str, **kwargs) -> Generator`
Stream audio synthesis in real-time.

#### `get_available_voices() -> List[Dict]`
Get list of available voices from ElevenLabs.

#### `set_voice_parameters(voice_id: str, **settings)`
Configure voice and synthesis settings.

#### `get_model_info() -> Dict`
Get information about the loaded model.

## Error Handling

### Common Issues

1. **API Key Not Found**
   ```
   Error: ElevenLabs API key is required
   ```
   Solution: Set the `ELEVENLABS_API_KEY` environment variable.

2. **Authentication Failed**
   ```
   Error: Invalid API key
   ```
   Solution: Verify your API key in the ElevenLabs dashboard.

3. **Quota Exceeded**
   ```
   Error: Character quota exceeded
   ```
   Solution: Check your usage limits in the ElevenLabs dashboard.

4. **Voice Not Found**
   ```
   Error: Voice ID not found
   ```
   Solution: Use `get_available_voices()` to find valid voice IDs.

### Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Examples

See the complete example script at `examples/elevenlabs_tts_example.py`:

```bash
# Run the example (requires API key)
python examples/elevenlabs_tts_example.py
```

## Pricing and Usage

- ElevenLabs charges based on character count
- Free tier includes limited characters per month
- Paid tiers offer higher quality voices and features
- Monitor usage in the ElevenLabs dashboard

## Integration with HoldTranscribe

### Configuration in HoldTranscribe

When using with HoldTranscribe, specify ElevenLabs as your TTS model:

```python
# In your HoldTranscribe configuration
tts_config = {
    "model_name": "eleven_multilingual_v2",
    "voice_id": "21m00Tcm4TlvDq8ikWAM",
    "api_key": os.getenv("ELEVENLABS_API_KEY")
}
```

### Real-time Usage

For real-time applications:

1. Use fast models (`eleven_turbo_v2_5`, `eleven_flash_v2_5`)
2. Enable streaming synthesis
3. Set `optimize_streaming_latency=3` or `4`
4. Use smaller chunk sizes for lower latency

## Support

- **ElevenLabs Documentation**: [docs.elevenlabs.io](https://docs.elevenlabs.io)
- **HoldTranscribe Issues**: Create an issue in the HoldTranscribe repository
- **ElevenLabs Support**: [help.elevenlabs.io](https://help.elevenlabs.io)

## License

The ElevenLabs integration follows the same license as HoldTranscribe. Note that ElevenLabs API usage is subject to their terms of service.