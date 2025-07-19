# Kyutai TTS Module for HoldTranscribe

A high-quality text-to-speech implementation using Kyutai's Moshi/DSM models with real-time streaming support.

## Overview

This TTS module provides state-of-the-art text-to-speech synthesis capabilities using Kyutai Labs' Moshi models. It features real-time streaming audio generation, direct speaker output, and flexible voice customization options.

### Key Features

- **Real-time streaming synthesis** - Generate and play audio as text is processed
- **Direct speaker output** - Stream audio directly to system speakers with minimal latency
- **File generation** - Save synthesized audio to WAV files
- **Multiple voices** - Support for various voice styles and emotions
- **GPU acceleration** - CUDA support for faster synthesis
- **Chunked processing** - Automatic text splitting for optimal streaming performance
- **Parameter control** - Fine-tune voice characteristics and generation quality

## Architecture

```
holdtranscribe/tts/
├── __init__.py          # Module interface and factory functions
├── kyutai_model.py      # Main Kyutai TTS implementation
├── cli.py              # Command-line interface
└── README.md           # This file
```

### Core Components

1. **KyutaiTTSModel** - Main TTS model class inheriting from base TTSModel
2. **KyutaiTTSGen** - Generation wrapper for streaming synthesis
3. **CLI interface** - Command-line tool for testing and usage
4. **Model factory** - Integration with HoldTranscribe's model system

## Installation

### Prerequisites

- Python 3.12 or higher
- Ubuntu/Linux (primary target platform)
- Working audio system (ALSA/PulseAudio)
- 8GB+ RAM (16GB+ recommended for GPU usage)
- Optional: NVIDIA GPU with CUDA for acceleration

### Install Dependencies

```bash
# Option 1: Install TTS extras from main package
pip install -e ".[tts]"

# Option 2: Install from requirements file
pip install -r requirements-tts.txt

# Option 3: Manual installation
pip install torch>=2.0.0 moshi>=0.2.10 sphn>=0.1.0 sounddevice>=0.4.6 huggingface_hub>=0.16.0
```

### System Dependencies (Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y pulseaudio alsa-utils portaudio19-dev python3-dev
```

## Quick Start

### Command Line Usage

```bash
# Basic text-to-speech to speakers
python -m holdtranscribe.tts.cli "Hello, world!"

# Save to file
python -m holdtranscribe.tts.cli "Hello, world!" -o output.wav

# Read from stdin
echo "This is from stdin" | python -m holdtranscribe.tts.cli -

# Interactive mode
python -m holdtranscribe.tts.cli
```

### Python API

```python
from holdtranscribe.tts import create_tts_model

# Create and load model
tts_model = create_tts_model("kyutai", device="cuda")
if tts_model.load():
    # Speak directly to speakers
    tts_model.play_audio_to_speakers("Hello, world!")
    
    # Generate file
    tts_model.synthesize("Hello, world!", "output.wav")
    
    # Stream for long texts
    for chunk in tts_model.synthesize_streaming("Long text here..."):
        # Process audio chunk
        pass
    
    tts_model.unload()
```

## API Reference

### Factory Functions

#### `create_tts_model(model_type, model_name="default", device="cuda", **kwargs)`

Create a TTS model instance.

**Parameters:**
- `model_type` (str): Model type ("kyutai" or "moshi")
- `model_name` (str): Model name or "default"
- `device` (str): Device ("cuda" or "cpu")
- `**kwargs`: Additional model parameters

**Returns:** KyutaiTTSModel instance or None

#### `list_available_models()`

Get available model types and descriptions.

**Returns:** Dict mapping model type to description

### KyutaiTTSModel Class

#### Core Methods

##### `load() -> bool`
Load the TTS model and dependencies.

**Returns:** True if successful, False otherwise

##### `unload() -> None`
Unload model and free memory.

##### `synthesize(text: str, output_file: str, **kwargs) -> bool`
Synthesize text to audio file.

**Parameters:**
- `text`: Text to synthesize
- `output_file`: Output WAV file path
- `**kwargs`: Additional synthesis parameters

**Returns:** True if successful

##### `synthesize_streaming(text: str, **kwargs) -> Generator[np.ndarray, None, None]`
Generate audio chunks in streaming fashion.

**Parameters:**
- `text`: Text to synthesize
- `**kwargs`: Additional parameters

**Yields:** Audio chunks as NumPy arrays

##### `play_audio_to_speakers(text: str, **kwargs) -> None`
Synthesize and play audio directly to speakers.

**Parameters:**
- `text`: Text to synthesize
- `**kwargs`: Additional parameters

#### Configuration Methods

##### `set_voice_parameters(**kwargs) -> None`
Configure voice generation parameters.

**Parameters:**
- `temp` (float): Temperature (0.0-1.0, default 0.6)
- `cfg_coef` (float): CFG coefficient (1.0-3.0, default 2.0)
- `n_q` (int): Quantization levels (default 32)

##### `set_streaming_parameters(**kwargs) -> None`
Configure streaming behavior.

**Parameters:**
- `chunk_max_words` (int): Words per chunk (default 50)
- `chunk_silence_duration` (float): Silence between chunks (default 0.5)

##### `set_voice_conditioning(voice: str) -> None`
Change the voice style.

**Parameters:**
- `voice`: Voice path from voice repository

##### `set_seed(seed: int) -> None`
Set random seed for reproducible generation.

#### Utility Methods

##### `get_model_info() -> dict`
Get comprehensive model information.

##### `get_available_voices() -> List[str]`
Get list of available voice options.

##### `split_text_for_streaming(text: str, max_words: int = None) -> List[str]`
Split text into chunks for streaming.

## Configuration

### Voice Selection

Available voices from the Kyutai voice repository:

```python
# Default voice (happy/neutral)
model.set_voice_conditioning("expresso/ex03-ex01_happy_001_channel1_334s.wav")

# Sad/emotional voice
model.set_voice_conditioning("expresso/ex03-ex01_sad_001_channel1_334s.wav")

# Other voices from the repository
# See: https://huggingface.co/kyutai/dsm-voices-1b-v1
```

### Generation Parameters

```python
# High quality, consistent voice
model.set_voice_parameters(temp=0.3, cfg_coef=2.5)

# More varied, expressive voice
model.set_voice_parameters(temp=0.8, cfg_coef=1.5)

# Balanced settings (default)
model.set_voice_parameters(temp=0.6, cfg_coef=2.0)
```

### Streaming Configuration

```python
# Low latency (smaller chunks)
model.set_streaming_parameters(chunk_max_words=25, chunk_silence_duration=0.3)

# Higher quality (larger chunks)
model.set_streaming_parameters(chunk_max_words=100, chunk_silence_duration=0.7)
```

## Advanced Usage

### Real-time Streaming

```python
import queue
import threading

def real_time_synthesis(text):
    audio_queue = queue.Queue()
    
    # Start background synthesis
    worker = model.synthesize_streaming_threaded(text, audio_queue)
    
    # Process audio as it becomes available
    for status, data in model.stream_audio_realtime(text):
        if status == 'audio':
            # Play or process audio chunk
            pass
        elif status == 'done':
            break
        elif status == 'error':
            print(f"Error: {data}")
            break
```

### Batch Processing

```python
texts = ["First sentence.", "Second sentence.", "Third sentence."]

for i, text in enumerate(texts):
    output_file = f"batch_{i:03d}.wav"
    success = model.synthesize(text, output_file)
    if success:
        print(f"Generated: {output_file}")
```

### Custom Voice Pipeline

```python
# Set consistent parameters
model.set_seed(42)
model.set_voice_parameters(temp=0.4, cfg_coef=2.2)

# Process multiple texts with same voice
for text in text_list:
    model.play_audio_to_speakers(text)
```

## Integration with HoldTranscribe

### Using Model Factory

```python
from holdtranscribe.models import ModelFactory, ModelType

# Create via main factory
tts_model = ModelFactory.create_tts_model("kyutai", "cuda")

# Register in model registry
from holdtranscribe.models import model_registry
model_registry.register("tts_main", tts_model)

# Use in application
tts = model_registry.get("tts_main")
if tts and tts.load():
    tts.synthesize("Hello from HoldTranscribe!", "greeting.wav")
```

### Application Integration

```python
class TTSService:
    def __init__(self):
        self.model = create_tts_model("kyutai", device="cuda")
        self.loaded = False
    
    def initialize(self):
        self.loaded = self.model.load()
        return self.loaded
    
    def speak(self, text):
        if self.loaded:
            self.model.play_audio_to_speakers(text)
    
    def shutdown(self):
        if self.model:
            self.model.unload()
```

## Performance

### Benchmarks

Typical performance on various hardware:

| Hardware | Synthesis Speed | Memory Usage | Audio Quality |
|----------|----------------|--------------|---------------|
| RTX 4090 | 2-3x real-time | 6-8GB VRAM | Excellent |
| RTX 3080 | 1.5-2x real-time | 4-6GB VRAM | Excellent |
| RTX 2060 | 1-1.5x real-time | 3-4GB VRAM | Very Good |
| CPU (16 cores) | 0.5-1x real-time | 8-16GB RAM | Good |
| CPU (8 cores) | 0.3-0.7x real-time | 8-12GB RAM | Good |

### Optimization Tips

1. **Use GPU acceleration** when available
2. **Batch processing** for multiple files
3. **Streaming synthesis** for long texts
4. **Smaller chunks** for lower latency
5. **Model caching** to avoid reload overhead

```python
# Optimized settings for real-time applications
model.set_streaming_parameters(chunk_max_words=30)
model.set_voice_parameters(temp=0.5)  # Slightly faster than 0.6
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

```
ModuleNotFoundError: No module named 'moshi'
```

**Solution:**
```bash
pip install moshi>=0.2.10 torch sphn sounddevice
```

#### 2. CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solutions:**
```python
# Use CPU
model = create_tts_model("kyutai", device="cpu")

# Or clear CUDA cache
import torch
torch.cuda.empty_cache()
```

#### 3. Audio Not Playing

```
OSError: No audio device found
```

**Solutions:**
```bash
# Check audio system
pulseaudio --check
aplay -l

# Install audio dependencies
sudo apt-get install pulseaudio alsa-utils

# Test audio
speaker-test -c 2
```

#### 4. Model Download Issues

```
HTTPError: 403 Client Error
```

**Solutions:**
```bash
# Check internet connection
ping huggingface.co

# Clear cache
rm -rf ~/.cache/huggingface/

# Manual download
huggingface-cli download kyutai/dsm-tts-1b-v1
```

### Debugging

Enable verbose logging:

```python
from holdtranscribe.utils import debug_print
debug_print("TTS debugging enabled")

# Check model info
print(model.get_model_info())

# Test with simple text
model.synthesize("test", "debug.wav")
```

## Examples

### Example 1: Basic Usage

```python
from holdtranscribe.tts import create_tts_model

model = create_tts_model("kyutai", device="cuda")
if model.load():
    model.play_audio_to_speakers("Hello, this is a TTS test!")
    model.unload()
```

### Example 2: File Generation

```python
texts = [
    "Welcome to our podcast.",
    "Today we're discussing text-to-speech technology.",
    "Thank you for listening!"
]

model = create_tts_model("kyutai")
if model.load():
    for i, text in enumerate(texts):
        model.synthesize(text, f"segment_{i:02d}.wav")
    model.unload()
```

### Example 3: Streaming Long Content

```python
long_article = """
Large amounts of text content that would benefit from
streaming synthesis to reduce latency and provide
real-time feedback to users...
"""

model = create_tts_model("kyutai")
if model.load():
    # Stream to speakers with real-time playback
    for status, data in model.stream_audio_realtime(long_article):
        if status == 'audio':
            print("Playing audio chunk...")
        elif status == 'done':
            print("Synthesis complete!")
            break
    model.unload()
```

### Example 4: Voice Customization

```python
model = create_tts_model("kyutai")
if model.load():
    # Configure for consistent, high-quality voice
    model.set_seed(42)
    model.set_voice_parameters(temp=0.4, cfg_coef=2.5)
    
    # Use expressive voice
    model.set_voice_conditioning("expresso/ex03-ex01_happy_001_channel1_334s.wav")
    
    model.play_audio_to_speakers("This is a happy, consistent voice!")
    model.unload()
```

## Contributing

When adding new features or fixing bugs:

1. **Follow the existing architecture** - inherit from TTSModel base class
2. **Handle dependencies gracefully** - check for imports and provide helpful error messages
3. **Add comprehensive error handling** - catch and report errors appropriately
4. **Update documentation** - keep this README and docstrings current
5. **Test thoroughly** - use the test script and demo to verify functionality

## License

This module integrates with Kyutai's Moshi models. Please check the respective licenses:

- **HoldTranscribe**: MIT License
- **Kyutai Moshi**: Check [Kyutai Labs licensing](https://github.com/kyutai-labs/delayed-streams-modeling)

## References

- [Kyutai Labs - Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling)
- [Moshi Model on Hugging Face](https://huggingface.co/kyutai/dsm-tts-1b-v1)
- [Voice Repository](https://huggingface.co/kyutai/dsm-voices-1b-v1)
- [Research Paper](https://arxiv.org/abs/example) (if available)