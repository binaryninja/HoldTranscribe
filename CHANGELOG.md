# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Future enhancements and features will be listed here

## [1.0.1] - 2025-07-18

### Added
- **Python Package Structure**: Complete restructure from standalone script to proper Python package
- **PyPI Installation**: Package can now be installed via `pip install holdtranscribe`
- **Global Command**: `holdtranscribe` command available system-wide after installation
- **Modern Packaging**: Added `pyproject.toml` with modern Python packaging standards
- **Setup Script**: Traditional `setup.py` for compatibility
- **Package Manifest**: `MANIFEST.in` for proper file inclusion
- **Installation Guide**: Comprehensive `INSTALL.md` with platform-specific instructions
- **Service Integration**: Complete systemd service setup documentation and examples
- **Cross-Platform Support**: Enhanced Windows, macOS, and Linux compatibility
- **Platform-Specific Hotkey Detection**: Improved hotkey handling across different platforms
- **GPU Acceleration**: Automatic CUDA detection with CPU fallback
- **Voice Activity Detection**: Smart audio capture using WebRTC VAD
- **Clipboard Integration**: Automatic text copying with cross-platform support
- **Desktop Notifications**: Visual feedback for transcription completion
- **Debug Mode**: Extensive logging and performance metrics
- **Model Flexibility**: Support for multiple Whisper model sizes (tiny to large-v3)
- **Performance Optimization**: Configurable beam search and fast mode options

### Changed
- **Command Interface**: Migrated from `python voice_hold_to_clip.py` to `holdtranscribe`
- **Project Structure**: Moved from single script to modular package structure
- **Installation Method**: Primary installation now via pip instead of manual script placement
- **Service Configuration**: Updated systemd/launchd configs for packaged installation
- **Documentation**: Complete rewrite of README and installation instructions
- **Path Handling**: Removed hardcoded paths in favor of standard package locations
- **Dependencies**: Organized dependencies with proper version constraints

### Removed
- **Standalone Script**: `voice_hold_to_clip.py` replaced by package entry point
- **Manual Installation**: Reduced emphasis on manual script copying
- **Hardcoded Paths**: Eliminated absolute paths in service configurations

### Fixed
- **Service Setup**: Corrected systemd service configuration for package installation
- **Cross-Platform Issues**: Resolved platform-specific path and permission problems
- **Documentation Accuracy**: Updated all examples to use new command structure
- **Installation Reliability**: Improved dependency resolution and package building

### Security
- **Input Validation**: Enhanced input handling for hotkey detection
- **Permission Management**: Improved service permissions and user group handling
- **Path Sanitization**: Removed hardcoded paths that could pose security risks

## [0.9.0] - 2025-07-17 (Pre-release)

### Added
- Initial standalone script implementation
- Basic Whisper integration
- Hotkey recording functionality
- Clipboard integration
- CUDA GPU support
- Voice Activity Detection
- Debug logging

### Notes
- This version existed as a standalone script (`voice_hold_to_clip.py`)
- Manual installation and setup required
- Limited cross-platform support
- No package management

---

## Installation

```bash
pip install holdtranscribe
```

## Usage

```bash
holdtranscribe --help
```

## Service Setup

### Linux (systemd)
```bash
# Create and enable service
systemctl --user enable holdtranscribe.service
systemctl --user start holdtranscribe.service
```

### macOS (launchd)
```bash
# Load launch agent
launchctl load ~/Library/LaunchAgents/com.holdtranscribe.plist
```

### Windows (Task Scheduler)
```cmd
# Create scheduled task
schtasks /create /tn "HoldTranscribe" /tr "holdtranscribe" /sc onlogon
```

## Links

- **Repository**: https://github.com/binaryninja/HoldTranscribe
- **Documentation**: https://github.com/binaryninja/HoldTranscribe#readme
- **Issue Tracker**: https://github.com/binaryninja/HoldTranscribe/issues
- **PyPI Package**: https://pypi.org/project/holdtranscribe/

## Contributing

We welcome contributions! Please see our [Contributing Guide](https://github.com/binaryninja/HoldTranscribe/blob/main/CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/binaryninja/HoldTranscribe/blob/main/LICENSE) file for details.