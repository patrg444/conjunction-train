#!/usr/bin/env python3
"""
Direct test for OpenSMILE audio feature extraction with proper error reporting
"""
import os
import sys
import time
import argparse
import numpy as np
import pyaudio
import threading
import subprocess
from subprocess import Popen, PIPE
import wave
import tempfile
import signal

def signal_handler(sig, frame):
    print("\nExiting...")
    sys.exit(0)

def list_audio_devices():
    """List all available audio input devices"""
    p = pyaudio.PyAudio()
    
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    
    devices = []
    
    for i in range(0, numdevices):
        device_info = p.get_device_info_by_index(i)
        name = device_info.get('name')
        inputs = device_info.get('maxInputChannels')
        
        # Only include input devices
        if inputs > 0:
            devices.append({
                'index': i,
                'name': name,
                'inputs': inputs
            })
    
    p.terminate()
    return devices

def record_audio(device_index, duration=3, rate=16000, chunk=1024, verbose=False):
    """Record audio from specified device and return as numpy array"""
    p = pyaudio.PyAudio()
    
    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=chunk
        )
        
        if verbose:
            print(f"Recording {duration} seconds from device {device_index}...")
        
        frames = []
        
        # Calculate total chunks to read
        total_chunks = int(rate / chunk * duration)
        
        for i in range(total_chunks):
            if verbose and i % 10 == 0:
                progress = i / total_chunks * 100
                sys.stdout.write(f"\rProgress: {progress:.1f}% ")
                sys.stdout.flush()
                
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)
        
        if verbose:
            print("\nFinished recording")
        
        stream.stop_stream()
        stream.close()
        
        # Convert to numpy array
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        
        return audio_data
        
    except Exception as e:
        print(f"Error recording audio: {str(e)}")
        return None
    finally:
        p.terminate()

def save_audio_to_wav(audio_data, filename, rate=16000):
    """Save numpy audio data to WAV file"""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes for int16
        wf.setframerate(rate)
        wf.writeframes(audio_data.tobytes())

def extract_opensmile_features(audio_data, opensmile_path, config_path, rate=16000, verbose=False):
    """Extract OpenSMILE features from audio data"""
    # Create a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_filename = temp_file.name
    
    try:
        # Save audio data to WAV
        save_audio_to_wav(audio_data, temp_filename, rate)
        
        if verbose:
            print(f"Temporary WAV file created: {temp_filename}")
            print(f"OpenSMILE path: {opensmile_path}")
            print(f"Config path: {config_path}")
        
        # Check if files exist
        if not os.path.exists(opensmile_path):
            print(f"ERROR: OpenSMILE executable not found at {opensmile_path}")
            return None
            
        if not os.path.exists(config_path):
            print(f"ERROR: OpenSMILE config not found at {config_path}")
            return None
            
        # Extract features with OpenSMILE
        cmd = [
            opensmile_path,
            "-C", config_path,
            "-I", temp_filename,
            "-csvoutput", "-",
            "-timestampcsv", "0",
            "-headercsv", "0"
        ]
        
        if verbose:
            print(f"Executing command: {' '.join(cmd)}")
        
        # Execute OpenSMILE
        process = Popen(cmd, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        
        # Check for errors
        if process.returncode != 0:
            print(f"OpenSMILE error: {stderr.decode('utf-8')}")
            return None
            
        if verbose:
            print("OpenSMILE executed successfully")
            
        # Parse the CSV output
        lines = stdout.decode('utf-8').strip().split('\n')
        features = []
        
        for line in lines:
            if line.strip():
                try:
                    values = [float(x) for x in line.split(';')]
                    features.append(values)
                except Exception as e:
                    print(f"Error parsing line: {line}")
                    print(f"Error: {str(e)}")
                    
        if not features:
            print("OpenSMILE returned no features")
            return None
            
        if verbose:
            print(f"Extracted {len(features)} feature frames, each with {len(features[0])} dimensions")
            
        return np.array(features)
        
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        return None
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)
            if verbose:
                print(f"Removed temporary file: {temp_filename}")

def find_opensmile():
    """Find OpenSMILE executable and config"""
    # Common paths
    possible_paths = [
        "./opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/bin/SMILExtract",
        "./opensmile-3.0.2-linux-x64/bin/SMILExtract",
        "./opensmile/bin/SMILExtract",
        "/usr/local/bin/SMILExtract"
    ]
    
    # Check each path
    for path in possible_paths:
        if os.path.exists(path):
            return path
            
    return None

def find_opensmile_config():
    """Find OpenSMILE config file"""
    # Common paths
    possible_paths = [
        "./opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/config/egemaps/v02/eGeMAPSv02.conf",
        "./opensmile-3.0.2-linux-x64/config/egemaps/v02/eGeMAPSv02.conf",
        "./opensmile/config/egemaps/v02/eGeMAPSv02.conf"
    ]
    
    # Check each path
    for path in possible_paths:
        if os.path.exists(path):
            return path
            
    return None

def continuous_audio_test(device_index, opensmile_path, config_path, verbose=False):
    """Continuously record and extract features"""
    print("Starting continuous audio test...")
    print("Press Ctrl+C to stop")
    
    rate = 16000
    chunk = 1024
    duration = 2  # seconds per recording
    
    try:
        while True:
            print("\n--- New recording ---")
            audio_data = record_audio(device_index, duration, rate, chunk, verbose)
            
            if audio_data is None:
                print("Failed to record audio, trying again...")
                time.sleep(1)
                continue
                
            # Check audio level
            rms = np.sqrt(np.mean(np.square(audio_data)))
            print(f"Audio RMS level: {rms:.2f}")
            
            # Extract features
            features = extract_opensmile_features(audio_data, opensmile_path, config_path, rate, verbose)
            
            if features is None:
                print("Failed to extract features, trying again...")
                time.sleep(1)
                continue
                
            print(f"Successfully extracted features: {features.shape}")
            
            # Short pause
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nStopping continuous test")

def main():
    parser = argparse.ArgumentParser(description='Test OpenSMILE feature extraction')
    parser.add_argument('--device', type=int, default=1, 
                        help='Audio device index to use')
    parser.add_argument('--duration', type=int, default=3, 
                        help='Duration to record in seconds')
    parser.add_argument('--opensmile', type=str, default=None, 
                        help='Path to OpenSMILE executable')
    parser.add_argument('--config', type=str, default=None, 
                        help='Path to OpenSMILE config file')
    parser.add_argument('--continuous', action='store_true', 
                        help='Run in continuous mode')
    parser.add_argument('--list-devices', action='store_true', 
                        help='List audio devices')
    parser.add_argument('--verbose', action='store_true', 
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # List devices if requested
    if args.list_devices:
        devices = list_audio_devices()
        print("\n=== Audio Input Devices ===")
        for device in devices:
            print(f"Device {device['index']}: {device['name']} ({device['inputs']} inputs)")
        return
    
    # Find OpenSMILE if not provided
    opensmile_path = args.opensmile or find_opensmile()
    if not opensmile_path:
        print("ERROR: Could not find OpenSMILE executable")
        return
        
    # Find config if not provided
    config_path = args.config or find_opensmile_config()
    if not config_path:
        print("ERROR: Could not find OpenSMILE config file")
        return
        
    print(f"Using OpenSMILE: {opensmile_path}")
    print(f"Using config: {config_path}")
    print(f"Using audio device: {args.device}")
    
    # Run in continuous mode if requested
    if args.continuous:
        continuous_audio_test(args.device, opensmile_path, config_path, args.verbose)
        return
    
    # Record audio
    print(f"Recording {args.duration} seconds from device {args.device}...")
    audio_data = record_audio(args.device, args.duration, verbose=args.verbose)
    
    if audio_data is None:
        print("Failed to record audio")
        return
    
    # Save audio to WAV for reference
    output_wav = "test_recording.wav"
    save_audio_to_wav(audio_data, output_wav)
    print(f"Audio saved to {output_wav}")
    
    # Extract features
    print("Extracting OpenSMILE features...")
    features = extract_opensmile_features(audio_data, opensmile_path, config_path, verbose=args.verbose)
    
    if features is None:
        print("Failed to extract features")
        return
        
    print(f"Successfully extracted features: {features.shape}")
    
    # Print sample of features
    if features.size > 0:
        print("\nSample of extracted features:")
        print(features[0][:10])  # First 10 features of first frame

if __name__ == "__main__":
    main()
