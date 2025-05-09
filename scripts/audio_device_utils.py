#!/usr/bin/env python
"""
Audio Device Utilities

Helper functions for selecting and configuring audio devices for the emotion recognition system.
This addresses specific issues with microphone selection and configuration.
"""

import pyaudio
import numpy as np
import time
import wave
import os
import tempfile
import subprocess
from subprocess import Popen, PIPE
import feature_normalizer

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
        outputs = device_info.get('maxOutputChannels')
        
        # Only include input devices
        if inputs > 0:
            devices.append({
                'index': i,
                'name': name,
                'inputs': inputs,
                'outputs': outputs
            })
    
    p.terminate()
    return devices

def find_best_microphone():
    """Find the best microphone device automatically"""
    devices = list_audio_devices()
    
    if not devices:
        return None
    
    # Look for likely microphone devices by name
    for device in devices:
        name = device['name'].lower()
        if any(keyword in name for keyword in ['mic', 'microphone']):
            return device['index']
    
    # If no specific microphone found, return the first input device
    return devices[0]['index']

def test_audio_device(device_index, duration=1, rate=16000):
    """
    Test if an audio device is working correctly.
    
    Args:
        device_index: The index of the audio device to test
        duration: Duration to record in seconds
        rate: Sample rate
        
    Returns:
        tuple: (success, audio_data, log_message)
    """
    chunk = 1024
    formats_to_try = [pyaudio.paInt16, pyaudio.paFloat32]
    
    for audio_format in formats_to_try:
        p = pyaudio.PyAudio()
        try:
            device_info = p.get_device_info_by_index(device_index)
            device_name = device_info.get('name')
            
            # Try to open and record from the device
            stream = p.open(
                format=audio_format,
                channels=1,
                rate=rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=chunk
            )
            
            frames = []
            for _ in range(0, int(rate / chunk * duration)):
                data = stream.read(chunk, exception_on_overflow=False)
                frames.append(data)
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # Convert the data to numpy array and check if it contains actual sound
            if audio_format == pyaudio.paFloat32:
                audio_data = np.frombuffer(b''.join(frames), dtype=np.float32)
            else:
                audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
            
            # Check if the audio has any significant content
            rms = np.sqrt(np.mean(np.square(audio_data)))
            if rms > 0.01 if audio_format == pyaudio.paFloat32 else 100:  # Adjust threshold based on format
                return True, audio_data, f"Device {device_index} ({device_name}) successfully captured audio data in format {audio_format}"
            
        except Exception as e:
            pass
            
        finally:
            try:
                if 'stream' in locals() and stream:
                    stream.stop_stream()
                    stream.close()
            except:
                pass
            p.terminate()
    
    # If all formats failed, return false
    return False, None, f"Failed to get valid audio from device {device_index}"

def find_working_microphone(fallback_device=None):
    """
    Find a working microphone by testing each one
    
    Args:
        fallback_device: Index to try first before scanning others
        
    Returns:
        tuple: (device_index, format, message)
    """
    # First test the fallback device if provided
    if fallback_device is not None:
        try:
            success, _, message = test_audio_device(fallback_device)
            if success:
                return fallback_device, pyaudio.paInt16, message
        except Exception as e:
            print(f"Error testing fallback device {fallback_device}: {str(e)}")
    
    # Try each audio input device
    devices = list_audio_devices()
    for device in devices:
        device_index = device['index']
        
        # Skip the fallback if we already tested it
        if device_index == fallback_device:
            continue
        
        try:
            success, _, message = test_audio_device(device_index)
            if success:
                return device_index, pyaudio.paInt16, message
        except Exception as e:
            print(f"Error testing device {device_index}: {str(e)}")
    
    # If no working device found, return the first device with a warning
    if devices:
        return devices[0]['index'], pyaudio.paInt16, f"WARNING: Could not verify device {devices[0]['index']} is working"
    
    # If no devices at all
    return 0, pyaudio.paInt16, "ERROR: No audio input devices found"

def get_best_audio_format(device_index):
    """Determine the best audio format for a device"""
    for fmt in [pyaudio.paInt16, pyaudio.paFloat32]:
        p = pyaudio.PyAudio()
        try:
            stream = p.open(
                format=fmt,
                channels=1,
                rate=16000,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=1024
            )
            stream.stop_stream()
            stream.close()
            p.terminate()
            return fmt
        except:
            p.terminate()
    
    # Default to int16 if no format works
    return pyaudio.paInt16

def extract_opensmile_features(audio_data, opensmile_path, config_path, audio_format=pyaudio.paInt16, rate=16000):
    """
    Extract OpenSMILE features from audio data
    
    Args:
        audio_data: Audio data as a numpy array
        opensmile_path: Path to OpenSMILE executable
        config_path: Path to OpenSMILE config file
        audio_format: Format of audio data
        rate: Sample rate
    
    Returns:
        np.array: OpenSMILE features for each frame (89 dimensions to match model) or None on failure
    """
    # Create a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_filename = temp_file.name
    
    try:
        # Get bytes data from numpy array
        if audio_format == pyaudio.paFloat32:
            byte_data = audio_data.astype(np.float32).tobytes()
        else:
            byte_data = audio_data.astype(np.int16).tobytes()
            
        # Write to WAV file
        with wave.open(temp_filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2 if audio_format == pyaudio.paInt16 else 4)
            wf.setframerate(rate)
            wf.writeframes(byte_data)
            
        # Extract features with OpenSMILE
        cmd = [
            opensmile_path,
            "-C", config_path,
            "-I", temp_filename,
            "-csvoutput", "-",
            "-timestampcsv", "0",
            "-headercsv", "0"
        ]
        
        # Debug the command
        cmd_str = " ".join(cmd)
        print(f"DEBUG: OpenSMILE command: {cmd_str}")
        
        # Execute OpenSMILE
        process = Popen(cmd, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        
        # Parse the CSV output
        if process.returncode != 0:
            print(f"OpenSMILE error: {stderr.decode('utf-8')}")
            # Print more details about the error
            with open(temp_filename, 'rb') as f:
                wav_size = len(f.read())
                print(f"DEBUG: WAV file size: {wav_size} bytes")
            return None
            
        lines = stdout.decode('utf-8').strip().split('\n')
        features = []
        
        for line in lines:
            if line.strip():
                values = [float(x) for x in line.split(';')]
                features.append(values)
                
        if not features:
            print("OpenSMILE returned no features")
            return None
        
        features_array = np.array(features)
        
        # Ensure the features have 89 dimensions (model requirement)
        # If OpenSMILE returns 88 features, pad with zeros to make 89
        if features_array.shape[1] == 88:
            print("Adding padding to make 89-dimensional features (model requirement)")
            padding = np.zeros((features_array.shape[0], 1))
            features_array = np.hstack((features_array, padding))
        elif features_array.shape[1] != 89:
            print(f"WARNING: Unexpected feature dimension: {features_array.shape[1]} (expected 89)")
        
        # Apply normalization to match what was used during training
        features_array = feature_normalizer.normalize_features(features_array, name="audio")
            
        return features_array
        
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        return None
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)
