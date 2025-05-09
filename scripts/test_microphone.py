#!/usr/bin/env python
"""
Simple microphone test to verify audio input works
"""

import pyaudio
import wave
import numpy as np
import time
import os

print("Starting microphone test...")

# Constants
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 3
OUTPUT_FILE = "test_audio.wav"

# Initialize PyAudio
p = pyaudio.PyAudio()

try:
    # Get device info
    info = p.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')
    
    print(f"Found {num_devices} audio devices")
    
    for i in range(0, num_devices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        print(f"Device {i}: {device_info['name']}")
        print(f"  Input channels: {device_info['maxInputChannels']}")
        print(f"  Output channels: {device_info['maxOutputChannels']}")
        print(f"  Default sample rate: {device_info['defaultSampleRate']}")
        
    # Open stream
    print("\nOpening stream...")
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    print("Recording for 3 seconds...")
    
    # Record audio
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        # Convert data to numpy array for analysis
        data_np = np.frombuffer(data, dtype=np.float32)
        max_val = np.max(np.abs(data_np))
        if i % 5 == 0:  # Print only every 5th chunk to avoid flooding
            print(f"Audio level: {max_val:.6f}")
    
    print("Finished recording")
    
    # Stop and close stream
    stream.stop_stream()
    stream.close()
    
    # Save to WAV file
    print(f"Saving to {OUTPUT_FILE}...")
    wf = wave.open(OUTPUT_FILE, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    print(f"File saved: {os.path.abspath(OUTPUT_FILE)}")
    
except Exception as e:
    print(f"Error in audio test: {str(e)}")
finally:
    # Terminate PyAudio
    p.terminate()
    print("PyAudio terminated")
