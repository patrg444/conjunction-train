#!/usr/bin/env python
"""
Test script for audio-video synchronization with proper audio input device
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import pyaudio
import threading
import queue
import wave
import tempfile
from subprocess import Popen, PIPE

# Global variables for synchronization
audio_buffer = []
video_buffer = []
audio_recording = True
lock = threading.Lock()

# Constants
RATE = 16000  # Audio sampling rate
CHUNK = 1024  # Audio chunk size
FPS = 15      # Target video FPS

def process_audio(audio_device, display_fn):
    """
    Process audio in a separate thread
    
    Args:
        audio_device: Audio device index
        display_fn: Function to call to display audio buffer size
    """
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    # List all audio devices
    print("\nAudio devices available:")
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        print(f"Device {i}: {dev_info['name']}")
        print(f"  Input channels: {dev_info['maxInputChannels']}")
        print(f"  Default sample rate: {dev_info['defaultSampleRate']}")
    
    # Open stream with specific device index
    try:
        print(f"\nAttempting to open audio device {audio_device}...")
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=RATE,
            input=True,
            input_device_index=audio_device,
            frames_per_buffer=CHUNK
        )
        print(f"Successfully opened audio device {audio_device}")
    except Exception as e:
        print(f"Error opening audio device {audio_device}: {str(e)}")
        print("Trying to open default audio input device...")
        try:
            stream = p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            print("Successfully opened default audio device")
        except Exception as e:
            print(f"Error opening default audio device: {str(e)}")
            return
    
    # Process audio continuously
    global audio_recording, audio_buffer
    frames_recorded = 0
    max_frames = FPS * 5  # 5 seconds worth at target FPS
    
    print("Audio recording thread started, collecting frames...")
    
    while audio_recording and frames_recorded < max_frames:
        try:
            # Read audio data
            data = stream.read(CHUNK, exception_on_overflow=False)
            
            # Convert to numpy array for level detection
            data_np = np.frombuffer(data, dtype=np.float32)
            level = np.max(np.abs(data_np))
            
            # Add frame to buffer
            with lock:
                audio_buffer.append((data, level))
                frames_recorded += 1
            
            # Update display
            display_fn(len(audio_buffer), level)
            
            # Short delay to not consume all CPU
            time.sleep(0.01)
            
        except Exception as e:
            print(f"Error reading audio: {str(e)}")
            break
    
    # Cleanup
    stream.stop_stream()
    stream.close()
    p.terminate()
    print(f"Audio thread finished, recorded {frames_recorded} frames")
    
    # Save audio buffer to file for verification
    if audio_buffer:
        try:
            wf = wave.open("test_audio_sync.wav", 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paFloat32))
            wf.setframerate(RATE)
            wf.writeframes(b''.join([frame[0] for frame in audio_buffer]))
            wf.close()
            print("Saved audio buffer to test_audio_sync.wav")
        except Exception as e:
            print(f"Error saving audio file: {str(e)}")

def capture_video(camera_index, display_fn):
    """
    Capture video frames
    
    Args:
        camera_index: Camera index
        display_fn: Function to call to display video buffer size
    """
    # Open camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return
    
    # Set FPS
    cap.set(cv2.CAP_PROP_FPS, FPS)
    
    # Create window
    window_name = "Audio-Video Sync Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    
    global video_buffer, audio_recording
    frames_captured = 0
    start_time = time.time()
    
    print("Video capture started...")
    
    # Main loop
    while audio_recording:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera")
            break
        
        # Add frame to buffer
        with lock:
            video_buffer.append(frame)
            frames_captured += 1
        
        # Get buffer sizes for display
        audio_size = len(audio_buffer)
        video_size = len(video_buffer)
        
        # Display info on frame
        cv2.putText(
            frame, f"Audio frames: {audio_size}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        
        cv2.putText(
            frame, f"Video frames: {frames_captured}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        
        if audio_size > 0:
            last_level = audio_buffer[-1][1]
            # Draw a level meter
            bar_width = int(last_level * 400)  # Scale the level to 400 pixels max
            cv2.rectangle(
                frame, (10, 100), (10 + bar_width, 130),
                (0, 255, 0), -1
            )
            
            cv2.putText(
                frame, f"Audio level: {last_level:.4f}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
        
        # Calculate and display FPS
        elapsed = time.time() - start_time
        if elapsed > 0:
            fps = frames_captured / elapsed
            cv2.putText(
                frame, f"FPS: {fps:.1f}",
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
        
        # Display frame
        cv2.imshow(window_name, frame)
        
        # Check for exit key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        
        # Update display function
        display_fn(video_size, frames_captured)
        
        # Control frame rate
        time.sleep(max(0, 1/FPS - (time.time() - start_time)/frames_captured))
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    audio_recording = False
    print(f"Video capture finished, captured {frames_captured} frames")

def update_display(audio_size, video_size):
    """
    Update display with buffer sizes (used for thread communication)
    """
    pass  # This function is just a placeholder for thread communication

def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description="Audio-Video Synchronization Test")
    
    parser.add_argument(
        "--camera_index", type=int,
        default=0,
        help="Camera index"
    )
    
    parser.add_argument(
        "--audio_device", type=int,
        default=1,  # Default to MacBook Pro Microphone
        help="Audio device index"
    )
    
    args = parser.parse_args()
    
    # Start audio thread
    audio_thread = threading.Thread(
        target=process_audio,
        args=(args.audio_device, update_display)
    )
    audio_thread.daemon = True
    audio_thread.start()
    
    # Run video capture (will block until finished)
    capture_video(args.camera_index, update_display)
    
    # Wait for audio thread to finish
    if audio_thread.is_alive():
        global audio_recording
        audio_recording = False
        audio_thread.join(timeout=2.0)
    
    print("Test completed")
    print(f"Final audio buffer size: {len(audio_buffer)}")
    print(f"Final video buffer size: {len(video_buffer)}")

if __name__ == "__main__":
    main()
