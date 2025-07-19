# ElevenLabs TTS Quick Start Guide

Get high-quality text-to-speech up and running in HoldTranscribe in under 5 minutes!

## Quick Setup (5 Steps)

### 1. Get Your ElevenLabs API Key
- Visit [elevenlabs.io](https://elevenlabs.io/app/settings/api-keys)
- Sign up (free tier: 10,000 characters/month)
- Generate an API key

### 2. Install ElevenLabs Package
```bash
pip install elevenlabs>=1.0.0
```

### 3. Set Your API Key
```bash
# Linux/macOS
export ELEVENLABS_API_KEY="your_api_key_here"

# Windows Command Prompt
set ELEVENLABS_API_KEY=your_api_key_here

# Windows PowerShell
$env:ELEVENLABS_API_KEY="your_api_key_here"
```

### 4. Test the Integration
```bash
python test_elevenlabs_integration.py
```

### 5. Run HoldTranscribe with TTS
```bash
# Basic TTS (ElevenLabs is now default)
python -m holdtranscribe.main --tts

# With AI Assistant + TTS
python -m holdtranscribe.main --model mistralai/Voxtral-Mini-3B-2507 --tts
```

## Usage Examples

### Basic Commands
```bash
# Default high-quality TTS
holdtranscribe --tts

# Fast TTS (lower latency)
holdtranscribe --tts --tts-model eleven_turbo_v2_5

# Fastest TTS (minimal latency)
holdtranscribe --tts --tts-model eleven_flash_v2_5

# AI Assistant with TTS
holdtranscribe --model mistralai/Voxtral-Mini-3B-2507 --tts
```

### Python API
```python
from holdtranscribe.models import ModelFactory
import os

# Create TTS model
tts = ModelFactory.create_tts_model(
    "default",  # Uses ElevenLabs
    "cpu", 
    api_key=os.getenv("ELEVENLABS_API_KEY")
)

if tts.load():
    # Generate speech
    tts.synthesize("Hello from ElevenLabs!", "output.mp3")
    tts.unload()
```

## Model Options

| Model | Speed | Quality | Best For |
|-------|-------|---------|----------|
| `eleven_multilingual_v2` | Medium | Excellent | Default, high quality |
| `eleven_turbo_v2_5` | Fast | Very Good | Real-time applications |
| `eleven_flash_v2_5` | Fastest | Good | Low latency needs |

## How It Works

1. **Hold hotkey** (Ctrl/Cmd + Mouse Back Button) for AI assistant
2. **Speak your question**
3. **Release hotkey**
4. **Get text response** copied to clipboard
5. **Hear audio response** via ElevenLabs TTS (if `--tts` enabled)

## Testing

### Quick Test
```bash
# Run comprehensive demo
ELEVENLABS_API_KEY="your_key" python demo_elevenlabs_integration.py

# Run example script
ELEVENLABS_API_KEY="your_key" python examples/elevenlabs_tts_example.py
```

### Manual Test
```bash
# Test API connection
python -c "
from elevenlabs import ElevenLabs
import os
client = ElevenLabs(api_key=os.getenv('ELEVENLABS_API_KEY'))
print('✅ API connection successful!')
"
```

## Troubleshooting

### Common Issues

**"API key not found"**
```bash
echo $ELEVENLABS_API_KEY  # Should show your key
```

**"Missing permissions"**
- Some API keys have limited permissions
- Core TTS functionality still works
- Voice selection/info may be restricted

**"Network connection failed"**
```bash
ping api.elevenlabs.io  # Test connectivity
```

**"Package not installed"**
```bash
pip install elevenlabs>=1.0.0
```

### Verification Commands
```bash
# Check package installation
python -c "import elevenlabs; print('✅ Package installed')"

# Check API key
python -c "import os; print('✅ API key:', bool(os.getenv('ELEVENLABS_API_KEY')))"

# Test HoldTranscribe integration
python -c "
from holdtranscribe.models import ModelFactory
model = ModelFactory.create_tts_model('default', 'cpu')
print('✅ Integration ready:', bool(model))
"
```

## Performance Tips

- **Use `eleven_turbo_v2_5`** for real-time applications
- **Use `eleven_flash_v2_5`** for minimal latency
- **Monitor usage** in your ElevenLabs dashboard
- **Cache audio** for repeated phrases

## What's Changed

✅ **ElevenLabs is now the default TTS** (replacing Moshi)  
✅ **Cloud-based synthesis** (no local GPU needed)  
✅ **High-quality voices** with natural speech  
✅ **Fast synthesis** with streaming support  
✅ **Easy setup** with just an API key  

## Support

- **Full documentation**: `TTS_SETUP.md`
- **Test integration**: `python test_elevenlabs_integration.py`
- **Demo script**: `python demo_elevenlabs_integration.py`
- **Example usage**: `python examples/elevenlabs_tts_example.py`

## Pricing

- **Free**: 10,000 characters/month
- **Starter**: $5/month for 30,000 characters  
- **Creator**: $22/month for 100,000 characters

See [ElevenLabs Pricing](https://elevenlabs.io/pricing) for current rates.

---

**That's it!** You now have high-quality text-to-speech working with HoldTranscribe using ElevenLabs. 🎉