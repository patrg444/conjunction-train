#!/usr/bin/env python
"""
Audio Device Testing Script

This script helps diagnose audio input device issues by:
1. Listing all available audio devices with their indices
2. Optionally testing a specific device by recording and playing back audio
3. Showing real-time audio levels to verify microphone responsiveness
"""

import pyaudio
import wave
import numpy as np
import argparse
import time
import os
from datetime import datetime
import sys
import matplotlib.pyplot as plt
from threading import Thread
import queue

def list_audio_devices():
    """List all available audio input and output devices"""
    p = pyaudio.PyAudio()
    
    print("\n=== AUDIO DEVICES DETECTED ===")
    print("{:<5} {:<50} {:<10} {:<10}".format("Index", "Name", "Inputs", "Outputs"))
    print("-" * 80)
    
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    
    for i in range(0, numdevices):
        device_info = p.get_device_info_by_index(i)
        name = device_info.get('name')
        inputs = device_info.get('maxInputChannels')
        outputs = device_info.get('maxOutputChannels')
        
        # Highlight potential microphones
        is_input = inputs > 0
        highlight = "*" if is_input else " "
        
        print("{:<5} {:<50} {:<10} {:<10} {}".format(
            i, name[:47] + "..." if len(name) > 50 else name, 
            inputs, outputs, highlight if is_input else ""))
    
    print("\n* Devices with input channels are potential microphones")
    p.terminate()

def test_audio_device(device_index, duration=5, rate=16000):
    """Record audio from the specified device and play it back"""
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    
    p = pyaudio.PyAudio()
    
    # Verify the device exists
    try:
        device_info = p.get_device_info_by_index(device_index)
        device_name = device_info.get('name')
        max_inputs = device_info.get('maxInputChannels')
        if max_inputs == 0:
            print(f"ERROR: Device {device_index} ({device_name}) has no input channels")
            p.terminate()
            return False
    except IOError:
        print(f"ERROR: Device with index {device_index} not found")
        p.terminate()
        return False
    
    filename = f"test_recording_device_{device_index}.wav"
    
    print(f"\nTesting device {device_index}: {device_name}")
    print(f"Recording {duration} seconds of audio...")
    
    # Create a queue for audio data
    q = queue.Queue()
    stop_event = False
    
    # Thread to calculate and display audio levels
    def audio_levels_thread():
        while not stop_event:
            if not q.empty():
                data = q.get()
                # Convert data to numpy array
                y = np.frombuffer(data, dtype=np.int16)
                # Calculate RMS level
                rms = np.sqrt(np.mean(np.square(y)))
                # Convert to dB
                if rms > 0:
                    db = 20 * np.log10(rms)
                else:
                    db = -100
                
                # Visualize audio level
                bars = int((db + 100) / 5)  # Scale to 0-20 bars
                sys.stdout.write("\r[" + "█" * min(bars, 20) + " " * (20 - min(bars, 20)) + 
                                f"] {db:.1f} dB ")
                sys.stdout.flush()
            else:
                time.sleep(0.01)
    
    # Start the level visualization thread
    level_thread = Thread(target=audio_levels_thread)
    level_thread.daemon = True
    level_thread.start()
    
    try:
        # Open stream for recording
        stream = p.open(format=format,
                        channels=channels,
                        rate=rate,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=chunk)
        
        print("Speak into the microphone...")
        frames = []
        
        for i in range(0, int(rate / chunk * duration)):
            try:
                data = stream.read(chunk, exception_on_overflow=False)
                frames.append(data)
                q.put(data)
            except IOError as e:
                print(f"\nError recording: {e}")
                break
        
        print(f"\nFinished recording. Saving to {filename}...")
        
        # Stop the stream
        stream.stop_stream()
        stream.close()
        
        # Save the recorded audio to a WAV file
        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        print("Recording saved. Playing back...")
        
        # Open stream for playback
        stream = p.open(format=format,
                        channels=channels,
                        rate=rate,
                        output=True)
        
        # Read the recorded file and play it
        wf = wave.open(filename, 'rb')
        data = wf.readframes(chunk)
        
        while data:
            stream.write(data)
            data = wf.readframes(chunk)
        
        stream.stop_stream()
        stream.close()
        wf.close()
        
        print(f"Playback finished. Test recording saved to {filename}")
        success = True
        
    except Exception as e:
        print(f"Error testing audio device: {e}")
        success = False
    
    # Stop the level visualization thread
    stop_event = True
    level_thread.join(timeout=0.5)
    
    p.terminate()
    return success

def monitor_audio_device(device_index, duration=30, rate=16000):
    """Monitor audio levels from the specified device for a longer period"""
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    
    p = pyaudio.PyAudio()
    
    # Verify the device exists
    try:
        device_info = p.get_device_info_by_index(device_index)
        device_name = device_info.get('name')
        max_inputs = device_info.get('maxInputChannels')
        if max_inputs == 0:
            print(f"ERROR: Device {device_index} ({device_name}) has no input channels")
            p.terminate()
            return False
    except IOError:
        print(f"ERROR: Device with index {device_index} not found")
        p.terminate()
        return False
    
    print(f"\nMonitoring device {device_index}: {device_name}")
    print(f"Showing audio levels for {duration} seconds. Press Ctrl+C to stop.")
    print("Speak into the microphone to see levels change...")
    
    max_level = -100
    min_level = 0
    
    try:
        # Open stream for monitoring
        stream = p.open(format=format,
                        channels=channels,
                        rate=rate,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=chunk)
        
        start_time = time.time()
        while time.time() - start_time < duration:
            try:
                data = stream.read(chunk, exception_on_overflow=False)
                # Convert data to numpy array
                y = np.frombuffer(data, dtype=np.int16)
                # Calculate RMS level
                rms = np.sqrt(np.mean(np.square(y)))
                # Convert to dB
                if rms > 0:
                    db = 20 * np.log10(rms)
                else:
                    db = -100
                
                # Update min/max
                max_level = max(max_level, db)
                if db > -100:  # Ignore silence
                    if min_level == 0:  # First valid reading
                        min_level = db
                    else:
                        min_level = min(min_level, db)
                
                # Visualize audio level
                bars = int((db + 100) / 5)  # Scale to 0-20 bars
                sys.stdout.write("\r[" + "█" * min(bars, 20) + " " * (20 - min(bars, 20)) + 
                                f"] {db:.1f} dB (Min: {min_level:.1f}, Max: {max_level:.1f})")
                sys.stdout.flush()
                
                time.sleep(0.01)
            except IOError as e:
                print(f"\nError monitoring: {e}")
                break
            except KeyboardInterrupt:
                print("\nMonitoring stopped by user")
                break
        
        print("\nMonitoring complete.")
        
        # Stop the stream
        stream.stop_stream()
        stream.close()
        
    except Exception as e:
        print(f"Error monitoring audio device: {e}")
    
    p.terminate()
    
    if max_level > -50:  # If we detected some sound
        print(f"Device {device_index} is working and receiving audio input.")
        print(f"Audio level range: {min_level:.1f} dB to {max_level:.1f} dB")
        return True
    else:
        print(f"Device {device_index} doesn't seem to be receiving audio.")
        print("Make sure the microphone is not muted and permissions are granted.")
        return False

def find_best_audio_device():
    """Try to automatically find the best audio input device"""
    p = pyaudio.PyAudio()
    
    print("\n=== Searching for the best audio input device ===")
    
    # Get all devices with input channels
    input_devices = []
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    
    for i in range(0, numdevices):
        device_info = p.get_device_info_by_index(i)
        if device_info.get('maxInputChannels') > 0:
            input_devices.append((i, device_info))
    
    if not input_devices:
        print("No input devices found!")
        p.terminate()
        return None
    
    print(f"Found {len(input_devices)} input devices.")
    
    # Look for likely microphone devices by name
    likely_devices = []
    for idx, device_info in input_devices:
        name = device_info.get('name').lower()
        # Check for common microphone keywords
        if any(keyword in name for keyword in ['mic', 'microphone', 'input', 'audio in']):
            likely_devices.append((idx, device_info))
    
    # If we found likely microphones, return the first one
    if likely_devices:
        best_device = likely_devices[0]
        print(f"Recommended device: {best_device[0]} - {best_device[1].get('name')}")
        p.terminate()
        return best_device[0]
    # Otherwise return the first input device
    elif input_devices:
        best_device = input_devices[0]
        print(f"Recommended device: {best_device[0]} - {best_device[1].get('name')}")
        p.terminate()
        return best_device[0]
    # If all else fails
    else:
        print("No suitable input devices found.")
        p.terminate()
        return None

def main():
    parser = argparse.ArgumentParser(description='Audio Device Testing Tool')
    parser.add_argument('--list', action='store_true', help='List all audio devices')
    parser.add_argument('--test', type=int, help='Test a specific audio device by index', default=None)
    parser.add_argument('--monitor', type=int, help='Monitor audio levels from a device', default=None)
    parser.add_argument('--duration', type=int, help='Duration for recording/monitoring in seconds', default=5)
    parser.add_argument('--find-best', action='store_true', help='Find the best audio input device')
    
    args = parser.parse_args()
    
    if args.list:
        list_audio_devices()
    
    if args.find_best:
        best_device = find_best_audio_device()
        if best_device is not None:
            print(f"\nTo test this device, run: python {sys.argv[0]} --test {best_device}")
            print(f"To monitor this device, run: python {sys.argv[0]} --monitor {best_device}")
    
    if args.test is not None:
        test_audio_device(args.test, args.duration)
    
    if args.monitor is not None:
        monitor_audio_device(args.monitor, args.duration)
    
    # If no arguments were provided, list the devices
    if not (args.list or args.test is not None or args.monitor is not None or args.find_best):
        list_audio_devices()
        print("\nUse one of the following options:")
        print(f"  To test a device:    python {sys.argv[0]} --test DEVICE_INDEX")
        print(f"  To monitor a device: python {sys.argv[0]} --monitor DEVICE_INDEX")
        print(f"  To find the best device: python {sys.argv[0]} --find-best")

if __name__ == "__main__":
    main()
