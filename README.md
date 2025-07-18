# HoldTranscribe

Hotkey-Activated Voice-to-Clipboard Transcriber

A lightweight tool that records audio while you hold a configurable hotkey, transcribes speech using OpenAI’s Whisper model, and copies the result to your clipboard.

---

## Features

* Hold-to-record using a customizable hotkey combination
* GPU acceleration with automatic CUDA detection and CPU fallback
* Instant copy of transcribed text to the clipboard
* Persistent model instance for low-latency transcription
* Configurable model size and beam search settings
* Detailed debug output and performance metrics
* Installer script for one-step setup and service registration
* Voice Activity Detection (VAD) for clean audio capture

---

## Requirements

* Python 3.8 or later
* Bash-compatible shell (for installer script)
* A CUDA-capable GPU (optional, for hardware acceleration)
* PulseAudio or equivalent on Linux
* Permissions to read input events (e.g., user in `input` group)

---

## Pip Installation & Service Setup Examples

After publishing to PyPI, users can install the package via `pip` and register the systemd service in one of two ways:

**1. Direct pip install from GitHub**

```
pip install git+https://github.com/binaryninja/holdtranscribe.git
```

Then install the user service file:

```
mkdir -p ~/.config/systemd/user
cp $(python3 -c "import site; print(site.getsitepackages()[0])")/holdtranscribe/holdtranscribe.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable holdtranscribe.service
systemctl --user start holdtranscribe.service
```

**2. Local package install plus `setup.py` entry point**

If working from a local clone:

```
cd holdtranscribe
pip install .
```

Register the installed entry point service:

```
# The package installs 'holdtranscribe' CLI entry point that points to the script
holdtranscribe --install-service
# Internally this copies holdtranscribe.service into ~/.config/systemd/user and enables it
```

---

## Manual Installation

If you prefer manual setup, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/binaryninja/holdtranscribe.git
   cd holdtranscribe
   ```
2. Install Python dependencies:

   ```bash
   pip install faster-whisper sounddevice pynput webrtcvad pyperclip notify2 numpy psutil
   ```
3. (Optional) Install PyTorch with CUDA support:

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

---

## Usage

Run the script with default settings:

```bash
./voice_hold_to_clip.py
```

Common options:

```bash
--model <size>       Whisper model size (tiny, base, small, medium, large-v3). Default: large-v3
--beam-size <n>      Beam search width (1 for fastest). Default: 5
--fast               Shorthand for `--model base --beam-size 1`
--debug              Enable verbose timing and resource metrics
--device <cpu|cuda>  Force CPU or GPU mode
```

Example for ultra-fast transcription:

```bash
./voice_hold_to_clip.py --model tiny --beam-size 1
```

---

## Configuration

### Hotkey

By default, the script listens for `Ctrl + Button9` (mouse forward). To change the hotkey, edit the `HOTKEY` set in the script:

```python
HOTKEY = {keyboard.Key.ctrl, mouse.Button.button9}
```

### Environment Variables

* `CUDA_VISIBLE_DEVICES` to restrict or disable GPU usage
* `TRANSFORMERS_CACHE` to customize model cache location
* `DISABLE_NOTIFY=1` to suppress desktop notifications

---

## Service Setup

The installer script can register a systemd user service. To do it manually:

1. Create the service directory:

   ```bash
   mkdir -p ~/.config/systemd/user
   ```
2. Create `~/.config/systemd/user/holdtranscribe.service` with:

   ```ini
   [Unit]
   Description=HoldTranscribe Voice Transcriber
   After=graphical-session.target

   [Service]
   ExecStart=$(which python) $(pwd)/voice_hold_to_clip.py --model large-v3 --beam-size 1
   Restart=always
   RestartSec=5
   Environment=DISPLAY=:0
   Environment=XDG_RUNTIME_DIR=/run/user/$UID
   WorkingDirectory=$(pwd)
   ```
3. Enable and start the service:

   ```bash
   systemctl --user daemon-reload
   systemctl --user enable holdtranscribe.service
   systemctl --user start holdtranscribe.service
   ```

To view logs:

```bash
journalctl --user -u holdtranscribe.service -f
```

---

## Troubleshooting

* **CUDA/cuDNN errors**: Verify your CUDA and cuDNN installation and that `libcudnn_ops.so` is on `LD_LIBRARY_PATH`.
* **Audio device errors**: List devices with:

  ```bash
  python -c "import sounddevice as sd; print(sd.query_devices())"
  ```
* **Permission denied on input events**: Add your user to the `input` group:

  ```bash
  sudo usermod -aG input $USER
  ```
* **Hotkey conflicts**: Ensure no other application captures the same combination; test with `--debug`.

---

## Contributing

Contributions, issues, and feature requests are welcome. Please open a pull request or issue on GitHub.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
