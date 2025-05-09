#!/usr/bin/env python3
# coding: utf-8 -*-
"""
Face Detection Demo with Smoothed Emotion Recognition

This script demonstrates face detection with OpenCV and shows the camera feed
along with smoothed emotion bars. It's a step toward the full emotion recognition
system using actual facial detection.
"""

import cv2
import numpy as np
import argparse
import time
from collections import deque

# Set up argument parser
parser = argparse.ArgumentParser(description='Face Detection Demo with Smoothed Emotions')
parser.add_argument('--fps', type=int, default=15,
                    help='Target FPS for video processing (default: 15fps)')
parser.add_argument('--display_width', type=int, default=1200,
                    help='Width of display window')
parser.add_argument('--display_height', type=int, default=700,
                    help='Height of display window')
parser.add_argument('--window_size', type=int, default=30,
                    help='Size of the smoothing window (default: 30 frames)')
args = parser.parse_args()

# Global variables
running = True

# Frame rate control
target_fps = args.fps
frame_interval = 1.0 / target_fps
last_frame_time = 0

# Emotion labels and colors
emotions = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']
emotion_colors = [
    (0, 0, 255),     # Anger: Red (BGR)
    (0, 140, 255),   # Disgust: Orange
    (0, 255, 255),   # Fear: Yellow
    (0, 255, 0),     # Happy: Green
    (255, 255, 0),   # Neutral: Cyan
    (255, 0, 0)      # Sad: Blue
]

# Create smoothing buffers for each emotion
smoothing_window_size = args.window_size
emotion_buffers = [deque(maxlen=smoothing_window_size) for _ in range(len(emotions))]

# Current raw values (simulated from face detection)
current_raw_values = [np.random.random() * 0.5 for _ in range(len(emotions))]

# Initialize the buffers
for i, buffer in enumerate(emotion_buffers):
    # Start with the same initial value in all buffer slots
    initial_value = current_raw_values[i]
    for _ in range(smoothing_window_size):
        buffer.append(initial_value)

# Load face detection model - using Haar cascade for simplicity
# This will be replaced with more advanced face detection in the full version
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    print("Loaded face detection model")
except Exception as e:
    print(f"Error loading face detection model: {e}")
    face_cascade = None

def main():
    """Main function to run the face detection demo."""
    global running, last_frame_time, emotion_buffers, current_raw_values, face_cascade

    # Try several webcam indices
    cap = None
    for cam_index in [0, 1, 2]:
        print(f"Trying camera index {cam_index}...")
        cap = cv2.VideoCapture(cam_index)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret:
                print(f"Successfully opened camera at index {cam_index}")
                break
            else:
                print(f"Camera at index {cam_index} opened but failed to read frame")
                cap.release()
                cap = None
        else:
            print(f"Could not open camera at index {cam_index}")
    
    if cap is None or not cap.isOpened():
        print("Error: Could not open any webcam.")
        return

    # Set camera properties - attempt to match our target FPS
    cap.set(cv2.CAP_PROP_FPS, target_fps)

    print(f"Starting face detection demo with target {target_fps} FPS")
    print(f"Window size for smoothing: {smoothing_window_size} frames")
    print("Press 'q' or ESC to quit")

    frame_count = 0  # For tracking processing rate
    faces_detected = 0  # Count of frames with faces detected
    total_frames = 0

    try:
        while running:
            # Timing for FPS control
            current_time = time.time()
            elapsed = current_time - last_frame_time

            # Sleep to maintain target FPS
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
                continue

            # Record the time we're processing this frame
            last_frame_time = current_time
            frame_count += 1
            total_frames += 1

            # Capture video frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            # Create a combined display frame
            display_height = 700
            display_width = 1200
            display = np.zeros((display_height, display_width, 3), dtype=np.uint8)
            
            # Convert frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces - the core face detection step
            face_detected = False
            if face_cascade is not None:
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                # If faces are detected, update the emotion values based on face properties
                if len(faces) > 0:
                    face_detected = True
                    faces_detected += 1
                    
                    # Draw rectangles around the faces
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        # Generate simulated emotion values based on face position
                        # This is just a placeholder - real emotion detection would use a trained model
                        # We're using face position and size to create somewhat meaningful variations
                        # in emotion values that change based on your facial movements
                        
                        # Normalize the position and size values
                        x_norm = x / frame.shape[1]  # horizontal position normalized
                        y_norm = y / frame.shape[0]  # vertical position normalized
                        size_norm = (w * h) / (frame.shape[0] * frame.shape[1])  # size normalized
                        
                        # Use these normalized values to create somewhat meaningful emotion values
                        # These are still random, but influenced by face position and size
                        anger_base = 0.2 + x_norm * 0.3  # More anger when face is to the right
                        disgust_base = 0.1 + y_norm * 0.4  # More disgust when face is lower
                        fear_base = 0.1 + (1-size_norm) * 0.5  # More fear with smaller faces
                        happy_base = 0.3 + (1-y_norm) * 0.5  # More happy when face is higher
                        neutral_base = 0.4 - abs(0.5-x_norm) * 0.5  # More neutral when centered
                        sad_base = 0.2 + (1-x_norm) * 0.3  # More sad when face is to the left
                        
                        # Add some random variation
                        current_raw_values = [
                            max(0, min(1, anger_base + np.random.normal(0, 0.05))),
                            max(0, min(1, disgust_base + np.random.normal(0, 0.05))),
                            max(0, min(1, fear_base + np.random.normal(0, 0.05))),
                            max(0, min(1, happy_base + np.random.normal(0, 0.05))),
                            max(0, min(1, neutral_base + np.random.normal(0, 0.05))),
                            max(0, min(1, sad_base + np.random.normal(0, 0.05)))
                        ]
                        
                        # If multiple faces, just use the first one for now
                        break
            
            # If no face is detected, slightly randomize the current values
            if not face_detected:
                for i in range(len(emotions)):
                    current_raw_values[i] = max(0, min(1, current_raw_values[i] + np.random.normal(0, 0.05)))
            
            # Update emotion buffers for smoothing
            for i in range(len(emotions)):
                emotion_buffers[i].append(current_raw_values[i])
            
            # Calculate smoothed emotion probabilities
            smoothed_probs = [sum(buffer) / len(buffer) for buffer in emotion_buffers]
            
            # Find dominant emotion
            dominant_emotion_idx = np.argmax(smoothed_probs)
            dominant_emotion = emotions[dominant_emotion_idx]
            dominant_color = emotion_colors[dominant_emotion_idx]
            
            # Add face detection statistics
            detection_rate = (faces_detected / total_frames) * 100 if total_frames > 0 else 0
            cv2.putText(frame, f"Face detection: {detection_rate:.1f}%", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add dominant emotion text
            cv2.putText(frame, f"Emotion: {dominant_emotion}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, dominant_color, 2)
            
            # Resize the frame to fit the left side of the display
            frame_height = display_height
            frame_width = int(frame.shape[1] * (frame_height / frame.shape[0]))
            frame_resized = cv2.resize(frame, (frame_width, frame_height))
            
            # Place the frame on the left side of the display
            if frame_width <= display_width // 2:
                display[0:frame_height, 0:frame_width] = frame_resized
            else:
                # If the frame is too wide, resize it further
                frame_width = display_width // 2
                frame_resized = cv2.resize(frame, (frame_width, frame_height))
                display[0:frame_height, 0:frame_width] = frame_resized
            
            # Draw emotions visualization on the right side
            # Add title for raw values
            cv2.putText(display, "Raw Values (from face)", (display_width//2 + 50, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        
            # Add title for smoothed values
            cv2.putText(display, f"Smoothed Values (Window={smoothing_window_size})", (display_width//2 + 50, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Draw graph for each emotion
            bar_height = 25
            bar_width = 300
            x_start = display_width//2 + 50
            y_start = 120
            
            for i, emotion in enumerate(emotions):
                # Draw emotion label
                cv2.putText(display, emotion, (x_start, y_start + i*60 + bar_height//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Draw raw value bar
                cv2.rectangle(display, (x_start + 100, y_start + i*60),
                             (x_start + 100 + bar_width, y_start + i*60 + bar_height),
                             (50, 50, 50), -1)
                
                filled_width = int(bar_width * current_raw_values[i])
                cv2.rectangle(display, (x_start + 100, y_start + i*60),
                             (x_start + 100 + filled_width, y_start + i*60 + bar_height),
                             emotion_colors[i], -1)
                
                # Draw raw value text
                raw_text = f"{current_raw_values[i]:.2f}"
                cv2.putText(display, raw_text, 
                           (x_start + 100 + bar_width + 10, y_start + i*60 + bar_height - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                # Draw smoothed value bar (below raw value)
                cv2.rectangle(display, (x_start + 100, y_start + i*60 + bar_height + 5),
                             (x_start + 100 + bar_width, y_start + i*60 + bar_height*2 + 5),
                             (50, 50, 50), -1)
                
                filled_width = int(bar_width * smoothed_probs[i])
                cv2.rectangle(display, (x_start + 100, y_start + i*60 + bar_height + 5),
                             (x_start + 100 + filled_width, y_start + i*60 + bar_height*2 + 5),
                             emotion_colors[i], -1)
                
                # Draw smoothed value text
                smooth_text = f"{smoothed_probs[i]:.2f}"
                cv2.putText(display, smooth_text, 
                           (x_start + 100 + bar_width + 10, y_start + i*60 + bar_height*2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Add performance info at the bottom
            fps_text = f"Processing: {1.0 / elapsed:.1f} FPS (Target: {args.fps})"
            cv2.putText(display, fps_text, (20, display.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add instruction
            cv2.putText(display, "Press 'q' or ESC to quit", 
                       (display_width - 250, display.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Resize display if needed
            if display.shape[1] != args.display_width or display.shape[0] != args.display_height:
                display = cv2.resize(display, (args.display_width, args.display_height))
            
            # Show the display
            cv2.imshow('Face Detection with Smoothed Emotions', display)
            
            # Check for exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                running = False
                break

    except Exception as e:
        print(f"Error in main loop: {e}")
        import traceback
        traceback.print_exc()

    finally:
        running = False
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("Demo stopped.")

if __name__ == "__main__":
    main()
