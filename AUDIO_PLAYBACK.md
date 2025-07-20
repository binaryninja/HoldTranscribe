# Audio Playback System

HoldTranscribe now features an improved audio playback system that plays TTS-generated audio directly to your default audio device instead of using external players.

## Features

### Direct Audio Device Playback
- **Real-time streaming**: Audio plays as it's generated (for supported TTS models)
- **Low latency**: No external player startup delays
- **Smooth playback**: Buffered streaming prevents audio dropouts
- **Automatic cleanup**: Temporary files are automatically removed

### Dual Playback Modes

#### 1. Streaming Mode (Preferred)
For TTS models that support `synthesize_streaming()`:
- Audio chunks are played as they arrive
- Minimal latency between text input and audio output
- Ideal for ElevenLabs TTS and other streaming-capable models

#### 2. File Mode (Fallback)
For traditional file-based TTS models:
- Audio file is generated first, then played with buffering
- Still much faster than external players
- Maintains compatibility with all TTS models

## Configuration

### Command Line Options

```bash
# Configure audio buffer size (samples per buffer)
--audio-buffer-size 1024    # Default: 1024

# Configure audio queue size (maximum buffered chunks)
--audio-queue-size 10       # Default: 10

# Configure playback timeout (seconds)
--audio-timeout 30.0        # Default: 30.0
```

### Tuning for Your System

#### For Choppy Audio
If you experience choppy or interrupted audio:

```bash
# Increase buffer size for more stable playback
--audio-buffer-size 2048

# Increase queue size for more buffering
--audio-queue-size 20
```

#### For Lower Latency
If you want the fastest possible response:

```bash
# Decrease buffer size for lower latency
--audio-buffer-size 512

# Decrease queue size for minimal buffering
--audio-queue-size 5
```

## Requirements

### Core Dependencies
```bash
pip install sounddevice>=0.4.6
pip install pydub>=0.25.0
pip install numpy>=1.21.0
```

### System Requirements
- **Linux**: ALSA or PulseAudio
- **macOS**: CoreAudio (built-in)
- **Windows**: WASAPI or DirectSound (built-in)

## How It Works

### Streaming Playback Flow
1. TTS model generates audio chunks in real-time
2. Each chunk is converted from MP3/WAV to numpy array
3. Audio data is queued in memory buffers
4. `sounddevice.OutputStream` plays buffered chunks continuously
5. Playback continues until all chunks are processed

### File Playback Flow
1. TTS model generates complete audio file
2. File is loaded and converted to numpy array
3. Audio data is split into buffer-sized chunks
4. Chunks are queued and played via `sounddevice.OutputStream`
5. Temporary file is deleted after playback

## Benefits Over External Players

| Feature | New System | External Players |
|---------|------------|------------------|
| **Startup Time** | ~0ms | 50-500ms |
| **Memory Usage** | Low (streaming) | High (full file) |
| **System Dependencies** | Minimal | Requires player installation |
| **File Cleanup** | Automatic | Manual or system-dependent |
| **Cross-platform** | Consistent | Varies by platform |
| **Streaming Support** | Yes | No |

## Troubleshooting

### "pydub not available" Error
```bash
pip install pydub
```

### "sounddevice not available" Error
```bash
pip install sounddevice
```

On Linux, you may also need:
```bash
sudo apt-get install portaudio19-dev  # Ubuntu/Debian
sudo yum install portaudio-devel      # CentOS/RHEL
```

### Audio Device Issues
Check available audio devices:
```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

### Choppy Audio Solutions
1. **Increase buffer size**: `--audio-buffer-size 2048`
2. **Increase queue size**: `--audio-queue-size 20`
3. **Check system load**: Close other audio applications
4. **Update drivers**: Ensure audio drivers are current

### No Audio Output
1. Check default audio device settings
2. Verify volume levels are not muted
3. Test with the audio test script: `python test_audio_playback.py`
4. Fall back to system player mode (automatic fallback)

## Testing

Run the audio test suite to verify functionality:

```bash
cd HoldTranscribe
python test_audio_playback.py
```

This will test:
- Direct audio playback
- Buffered audio playback  
- Streaming audio simulation
- Audio device detection

## Advanced Configuration

### Environment Variables
```bash
# Force fallback to system players
export HOLDTRANSCRIBE_FORCE_SYSTEM_AUDIO=1

# Set custom audio device
export SOUNDDEVICE_DEFAULT_DEVICE=1
```

### Python API
```python
from holdtranscribe.app import HoldTranscribeApp

app = HoldTranscribeApp()
app.audio_buffer_size = 2048    # Custom buffer size
app.audio_queue_size = 15       # Custom queue size
app.audio_timeout = 45.0        # Custom timeout
```

## Performance Notes

- **Streaming Mode**: ~10-50ms latency
- **File Mode**: ~100-200ms latency
- **Memory Usage**: ~2-10MB during playback
- **CPU Usage**: ~1-5% during playback

The new system is designed to be lightweight while providing high-quality audio playback suitable for real-time AI interactions.