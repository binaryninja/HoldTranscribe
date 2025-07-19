# HoldTranscribe Architecture Documentation

## Overview

HoldTranscribe has been refactored from a monolithic structure into a modular, extensible architecture. This document describes the new architecture, its components, and how to work with and extend the system.

## Architecture Goals

The refactoring was designed to achieve:

1. **Modularity**: Separate concerns into distinct, well-defined modules
2. **Extensibility**: Easy to add new models (transcription, assistant, TTS)
3. **Maintainability**: Clear interfaces and separation of responsibilities
4. **Testability**: Each component can be tested independently
5. **Flexibility**: Support for different model implementations and configurations

## Directory Structure

```
holdtranscribe/
├── __init__.py
├── main.py                 # Backward compatibility entry point
├── app.py                  # Main application class
├── models/                 # Model interfaces and implementations
│   ├── __init__.py        # Base classes and factory
│   ├── whisper_model.py   # Whisper transcription implementation
│   ├── voxtral_model.py   # Voxtral assistant implementation
│   └── dia_model.py       # Dia TTS implementation
├── audio/                  # Audio processing and recording
│   └── __init__.py        # Audio recording, VAD, utilities
├── input/                  # Input handling (keyboard/mouse)
│   └── __init__.py        # Hotkey management, event handling
└── utils/                  # Utility functions
    └── __init__.py        # Debug, device detection, platform utils
```

## Core Components

### 1. Models Module (`models/`)

The models module provides a unified interface for different AI models through abstract base classes and a factory pattern.

#### Base Classes

- **`BaseModel`**: Abstract base for all models
- **`TranscriptionModel`**: Base for transcription models
- **`AssistantModel`**: Base for AI assistant models  
- **`TTSModel`**: Base for text-to-speech models

#### Factory Pattern

The `ModelFactory` class creates appropriate model instances based on configuration:

```python
from holdtranscribe.models import ModelFactory, ModelType

# Create models using the factory
transcription_model = ModelFactory.create_transcription_model("large-v3", "cuda")
assistant_model = ModelFactory.create_assistant_model("mistralai/Voxtral-Mini-3B-2507", "cuda")
tts_model = ModelFactory.create_tts_model("facebook/mms-tts-eng", "cuda")
```

#### Model Registry

The `ModelRegistry` manages loaded models:

```python
from holdtranscribe.models import model_registry

# Register models
model_registry.register("transcription", transcription_model)
model_registry.register("assistant", assistant_model)

# Access models
transcription_model = model_registry.get("transcription")

# Cleanup
model_registry.clear()  # Unloads all models
```

### 2. Audio Module (`audio/`)

Handles audio recording, voice activity detection (VAD), and audio processing utilities.

#### Key Classes

- **`AudioRecorder`**: Main recording interface with VAD
- **`VADProcessor`**: Voice activity detection using WebRTC VAD
- **`AudioBuffer`**: Thread-safe audio frame buffer
- **`AudioUtils`**: Utility functions for audio processing

#### Usage Example

```python
from holdtranscribe.audio import AudioRecorder, AudioUtils

# Create recorder
recorder = AudioRecorder(sample_rate=16000, vad_aggressiveness=2)

# Record audio
recorder.start_recording()
# ... wait for audio input ...
frames, stats = recorder.stop_recording()

# Convert to file
AudioUtils.frames_to_wav_file(frames, 16000, "output.wav")
```

### 3. Input Module (`input/`)

Manages keyboard and mouse input events, including hotkey combinations.

#### Key Classes

- **`InputManager`**: Main input coordination
- **`HotkeyManager`**: Hotkey combination management
- **`MouseManager`**: Mouse event handling

#### Usage Example

```python
from holdtranscribe.input import InputManager, InputMode

input_manager = InputManager()

# Register hotkeys
input_manager.register_hotkey(
    InputMode.TRANSCRIBE, 
    "ctrl+shift+t",
    on_press=start_transcription,
    on_release=stop_transcription
)

# Start listening
input_manager.start_listening()
```

### 4. Utils Module (`utils/`)

Provides utility functions for debugging, device detection, and platform-specific operations.

#### Key Functions

- **`detect_device()`**: Auto-detect CUDA GPU or fallback to CPU
- **`get_platform_hotkey()`**: Get platform-specific hotkey configurations
- **`debug_print()`**: Conditional debug logging
- **`get_memory_usage()`**: Memory usage monitoring

## Application Flow

### 1. Initialization

```
HoldTranscribeApp.__init__()
├── Parse command line arguments
├── Setup debug mode
├── Detect compute device (CUDA/CPU)
├── Initialize models via ModelFactory
├── Setup audio recording system
└── Configure input handling
```

### 2. Runtime Operation

```
Input Event → HotkeyManager → AudioRecorder → Model Processing → Output
     ↓             ↓              ↓              ↓            ↓
  Key/Mouse    Combination    VAD + Record    Transcribe/  Clipboard +
   Press       Detection      Audio Frames    Assistant    Notification
```

### 3. Model Processing Pipeline

```
Audio Frames → Temporary WAV → Model.process() → Text Output
     ↓              ↓              ↓              ↓
  Raw Audio    File Creation   Model-Specific   Formatted
   Buffer       (if needed)     Processing      Response
```

## Extending the System

### Adding New TTS Models

The refactored architecture makes it easy to add new TTS models. Here's how:

1. **Create a new TTS model class**:

```python
# holdtranscribe/models/my_tts_model.py
from . import TTSModel

class MyTTSModel(TTSModel):
    def __init__(self, model_name: str, device: str):
        super().__init__(model_name, device)
        # Initialize your model specifics
    
    def load(self) -> bool:
        # Load your TTS model
        try:
            # Model loading logic
            self.is_loaded = True
            return True
        except Exception as e:
            return False
    
    def unload(self):
        # Cleanup model resources
        pass
    
    def synthesize(self, text: str, output_file: str, **kwargs) -> bool:
        # Implement text-to-speech synthesis
        # Return True if successful
        pass
```

2. **Register in the factory**:

```python
# Update holdtranscribe/models/__init__.py
@staticmethod
def create_tts_model(model_name: str, device: str, **kwargs) -> Optional[TTSModel]:
    from .my_tts_model import MyTTSModel
    
    if "my_model_identifier" in model_name.lower():
        return MyTTSModel(model_name, device, **kwargs)
    # ... existing conditions
```

3. **Use the new model**:

```bash
python -m holdtranscribe.main --tts --tts-model "my_model_identifier"
```

### Adding New Transcription Models

Similar pattern for transcription models:

```python
class MyTranscriptionModel(TranscriptionModel):
    def transcribe(self, audio_data: Any, **kwargs) -> str:
        # Implement transcription logic
        pass
```

### Adding New Assistant Models

For AI assistant models:

```python
class MyAssistantModel(AssistantModel):
    def generate_response(self, audio_data: Any, prompt: str = None, **kwargs) -> str:
        # Implement response generation
        pass
```

## Configuration and Customization

### Model Configuration

Models can be configured through the factory:

```python
# Custom configuration for models
transcription_model = ModelFactory.create_transcription_model(
    "large-v3",
    "cuda",
    fast_mode=True,
    beam_size=1,
    custom_param="value"
)
```

### Audio Configuration

Audio recording can be customized:

```python
recorder = AudioRecorder(
    sample_rate=48000,          # Higher quality
    frame_duration_ms=20,       # Shorter frames
    vad_aggressiveness=3,       # More aggressive VAD
    channels=2                  # Stereo (if supported)
)
```

### Input Configuration

Hotkeys can be platform-specific:

```python
# Platform-specific hotkey setup
if platform.system() == "Darwin":  # macOS
    hotkey = "cmd+shift+t"
else:  # Windows/Linux
    hotkey = "ctrl+shift+t"
```

## Performance Considerations

### Memory Management

- Models are automatically unloaded when the application exits
- Use `model_registry.clear()` to explicitly unload models
- Monitor memory usage with `get_memory_usage()`

### GPU Utilization

- Automatic CUDA detection with fallback to CPU
- Models are moved to the detected device automatically
- Use `torch.cuda.empty_cache()` for GPU memory cleanup

### Audio Processing

- VAD reduces processing load by filtering silence
- Frame-based processing minimizes memory usage
- Temporary files are automatically cleaned up

## Error Handling

### Model Loading Errors

Models gracefully fallback when dependencies are missing:

```python
try:
    model = ModelFactory.create_tts_model("some_model", "cuda")
    if model and model.load():
        # Use model
    else:
        # Handle fallback
except ImportError:
    # Dependencies not available
```

### Audio Errors

Audio failures are handled gracefully:

```python
if not recorder.start_recording():
    print("Failed to start recording")
    # Handle error condition
```

## Testing

### Unit Testing

Each module can be tested independently:

```python
# Test model creation
def test_model_factory():
    model = ModelFactory.create_transcription_model("base", "cpu")
    assert model is not None
    assert model.load()

# Test audio recording
def test_audio_recording():
    recorder = AudioRecorder()
    assert recorder.start_recording()
    # Test recording functionality
```

### Integration Testing

Test complete workflows:

```python
def test_transcription_pipeline():
    # Setup models
    # Record audio
    # Process transcription
    # Verify output
```

## Migration Guide

### From Old Structure

The old monolithic `main.py` is preserved for backward compatibility. To migrate:

1. **Replace direct function calls** with model interfaces:
   ```python
   # Old
   text = whisper_model.transcribe(audio_file)
   
   # New
   transcription_model = model_registry.get("transcription")
   text = transcription_model.transcribe(audio_file)
   ```

2. **Use factory pattern** for model creation:
   ```python
   # Old
   model = load_whisper_model(device, fast_mode, beam_size)
   
   # New
   model = ModelFactory.create_transcription_model("large-v3", device, fast_mode=fast_mode)
   ```

3. **Update audio handling**:
   ```python
   # Old
   start_stream()
   audio_data = stop_stream()
   
   # New
   recorder.start_recording()
   frames, stats = recorder.stop_recording()
   ```

## Best Practices

### Model Development

1. Always inherit from the appropriate base class
2. Implement proper error handling in `load()` and processing methods
3. Include resource cleanup in `unload()`
4. Use type hints for better IDE support

### Configuration

1. Use factory pattern for model creation
2. Validate configuration parameters
3. Provide sensible defaults
4. Document configuration options

### Error Handling

1. Use graceful degradation when models fail to load
2. Provide informative error messages
3. Clean up resources in error conditions
4. Log errors for debugging

## Future Enhancements

### Planned Features

1. **Plugin System**: Load models dynamically from external packages
2. **Configuration Files**: YAML/JSON configuration for complex setups
3. **Model Caching**: Cache loaded models between sessions
4. **Streaming Processing**: Real-time processing for live applications
5. **Multi-language Support**: Automatic language detection and switching

### Extension Points

1. **Custom Audio Sources**: Support for network streams, files
2. **Output Formats**: Support for different output formats (JSON, XML)
3. **Custom Processors**: Add post-processing steps
4. **Integration APIs**: REST API for external integrations

This architecture provides a solid foundation for current needs while remaining flexible for future enhancements.