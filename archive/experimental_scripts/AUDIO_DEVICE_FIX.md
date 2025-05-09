# Audio Device Fix for Real-time Emotion Recognition

This document explains the issues with audio device handling in the real-time emotion recognition system and provides solutions.

## The Problem

The original code was using a hardcoded audio device index (1), which doesn't work reliably across different systems and setups. This resulted in:

1. No audio being captured ("audio buffer 0 frames" message)
2. The system showing "waiting" instead of emotion predictions
3. No actual emotion probabilities being displayed

## Solutions

We've implemented several fixes and improvements:

### 1. Audio Device Testing Utility

The `scripts/test_audio_devices.py` script allows you to:
- List all available audio input devices
- Test a specific device to see if it can capture audio
- Monitor audio levels
- Find the best microphone automatically

```bash
# List all audio devices
python scripts/test_audio_devices.py --list

# Test a specific device
python scripts/test_audio_devices.py --test 1

# Find the best device
python scripts/test_audio_devices.py --find-best
```

### 2. Audio Device Utilities Library

The `scripts/audio_device_utils.py` module provides robust functions for:
- Listing audio devices
- Finding the best microphone
- Testing audio devices
- Getting the best audio format
- Extracting audio features safely

### 3. Manual Device Selection Launcher

The `run_emotion_with_device.sh` script lets you manually specify which audio device to use:

```bash
# Run with default device (1)
./run_emotion_with_device.sh

# Run with a specific device (e.g., 0)
./run_emotion_with_device.sh 0

# Run with a different device (e.g., 4)
./run_emotion_with_device.sh 4
```

This allows for easy testing of different microphones.

### 4. Enhanced Emotion Recognition

The `scripts/enhanced_compatible_realtime_emotion.py` and `run_enhanced_emotion_sync.sh` provide:
- Auto-detection of working microphones
- Better error handling and status reporting
- Improved visualization of emotion probabilities
- Graceful fallback options

## How to Use

1. **First, find your working microphone**:
   ```bash
   python scripts/test_audio_devices.py --list
   ```

2. **Test specific microphones**:
   ```bash
   python scripts/test_audio_devices.py --test DEVICE_INDEX
   ```

3. **Run with a specific microphone**:
   ```bash
   ./run_emotion_with_device.sh DEVICE_INDEX
   ```

4. **Or use the enhanced auto-detection version**:
   ```bash
   ./run_enhanced_emotion_sync.sh
   ```

## Troubleshooting

If you continue to have issues:

1. **Check microphone permissions**:
   Make sure your system has granted microphone access to Python/terminal applications.

2. **Try different formats**:
   The utilities will automatically try multiple audio formats (Int16, Float32).

3. **Check audio levels**:
   Use the monitoring tool to check if your microphone is capturing audio:
   ```bash
   python scripts/test_audio_devices.py --monitor DEVICE_INDEX --duration 10
   ```

4. **Verify OpenSMILE**:
   Ensure OpenSMILE is properly installed and its path is correct in the launcher scripts.

## Technical Details

The key improvements include:

1. **Robust error handling** in audio device initialization and stream reading
2. **Multiple format support** to handle different audio hardware
3. **Automatic fallback mechanisms** when preferred devices fail
4. **Real-time status feedback** about audio processing
5. **Diagnostic tools** to isolate and fix specific issues
