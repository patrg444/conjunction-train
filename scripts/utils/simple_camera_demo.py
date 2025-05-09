#!/usr/bin/env python3
# coding: utf-8 -*-
"""
Simple Camera Demo for Emotion Recognition

This script demonstrates basic camera functionality with OpenCV,
showing what would be the foundation for the emotion recognition system.
"""

import cv2
import numpy as np
import argparse
import time

# Set up argument parser
parser = argparse.ArgumentParser(description='Simple Camera Demo')
parser.add_argument('--fps', type=int, default=15,
                    help='Target FPS for video processing (default: 15fps)')
parser.add_argument('--display_width', type=int, default=800,
                    help='Width of display window')
parser.add_argument('--display_height', type=int, default=600,
                    help='Height of display window')
args = parser.parse_args()

# Global variables
running = True

# Frame rate control
target_fps = args.fps
frame_interval = 1.0 / target_fps
last_frame_time = 0

def main():
    """Main function to run the camera demo."""
    global running, last_frame_time

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

    print(f"Starting camera demo with target {target_fps} FPS")
    print("Press 'q' or ESC to quit")

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

            # Capture video frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            # Create display frame
            display = frame.copy()

            # Simulate what the emotion recognition would display
            # Add placeholder for emotion display
            cv2.putText(display, "Emotion: <would show emotion here>", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add placeholder for confidence
            cv2.putText(display, "Confidence: <would show confidence here>", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Add FPS counter
            fps_text = f"Processing: {1.0 / elapsed:.1f} FPS (Target: {args.fps})"
            cv2.putText(display, fps_text, (20, display.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Bar placeholders for emotions
            emotions = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']
            bar_height = 20
            bar_width = 200
            x_start = 20
            y_start = 120

            for i, emotion in enumerate(emotions):
                # Draw emotion label
                cv2.putText(display, emotion, (x_start, y_start + i*40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Draw empty bar
                cv2.rectangle(display, (x_start, y_start + i*40 + 5),
                              (x_start + bar_width, y_start + i*40 + 5 + bar_height),
                              (100, 100, 100), -1)
                
                # Draw example filled portion (random value)
                example_value = np.random.random() * 0.8  # Random values between 0 and 0.8
                filled_width = int(bar_width * example_value)
                
                # Different colors for different emotions
                color = (
                    (0, 0, 255),     # Anger: Red (BGR)
                    (0, 140, 255),   # Disgust: Orange
                    (0, 255, 255),   # Fear: Yellow
                    (0, 255, 0),     # Happy: Green
                    (255, 255, 0),   # Neutral: Cyan
                    (255, 0, 0)      # Sad: Blue
                )[i]
                
                cv2.rectangle(display, (x_start, y_start + i*40 + 5),
                             (x_start + filled_width, y_start + i*40 + 5 + bar_height),
                             color, -1)

            # Resize display for better viewing
            display = cv2.resize(display, (args.display_width, args.display_height))

            # Show frame
            cv2.imshow('Camera Demo (Emotion Recognition Placeholder)', display)

            # Check for exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                running = False
                break

    except Exception as e:
        print(f"Error in main loop: {e}")

    finally:
        running = False
        cap.release()
        cv2.destroyAllWindows()
        print("Camera demo stopped.")

if __name__ == "__main__":
    main()
