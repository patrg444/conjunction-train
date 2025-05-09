#!/usr/bin/env python3
# coding: utf-8 -*-
"""
Smoothing Demonstration with Side-by-Side Comparison

This script clearly demonstrates the moving average smoothing effect by showing
both raw and smoothed emotion bar graphs side by side.
"""

import cv2
import numpy as np
import argparse
import time
from collections import deque

# Set up argument parser
parser = argparse.ArgumentParser(description='Smoothing Comparison Demo')
parser.add_argument('--fps', type=int, default=15,
                    help='Target FPS for video processing (default: 15fps)')
parser.add_argument('--display_width', type=int, default=1200,
                    help='Width of display window')
parser.add_argument('--display_height', type=int, default=700,
                    help='Height of display window')
parser.add_argument('--window_size', type=int, default=30,
                    help='Size of the smoothing window (default: 30 frames)')
parser.add_argument('--change_rate', type=float, default=0.3, 
                    help='Rate of random changes (0-1, higher = more jumpy)')
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

# Current raw values (for demonstration)
current_raw_values = [np.random.random() * 0.5 for _ in range(len(emotions))]

# Initialize the buffers
for i, buffer in enumerate(emotion_buffers):
    # Start with the same initial value in all buffer slots
    initial_value = current_raw_values[i]
    for _ in range(smoothing_window_size):
        buffer.append(initial_value)

def main():
    """Main function to run the comparison demo."""
    global running, last_frame_time, emotion_buffers, current_raw_values

    # Create a plain black canvas to display the bars
    height = 700
    width = 1200
    
    print(f"Starting smoothing comparison demo with target {target_fps} FPS")
    print(f"Window size: {smoothing_window_size} frames")
    print(f"Change rate: {args.change_rate} (higher = more jumpy)")
    print("Press 'q' or ESC to quit")

    frame_count = 0  # For tracking processing rate

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

            # Create black canvas
            display = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Update raw emotion values - using the specified change rate
            for i in range(len(emotions)):
                # Apply random walk with the specified change rate
                change = (np.random.random() - 0.5) * args.change_rate
                current_raw_values[i] = max(0, min(1, current_raw_values[i] + change))
                
                # Add to buffer for smoothing
                emotion_buffers[i].append(current_raw_values[i])

            # Calculate smoothed emotion probabilities
            smoothed_probs = [sum(buffer) / len(buffer) for buffer in emotion_buffers]
            
            # Add title for raw values
            cv2.putText(display, "Raw Values (Random Walk)", (150, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
            # Add title for smoothed values
            cv2.putText(display, f"Smoothed Values (Moving Avg, Window={smoothing_window_size})", (650, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Add FPS counter
            fps_text = f"Processing: {1.0 / elapsed:.1f} FPS (Target: {args.fps})"
            cv2.putText(display, fps_text, (20, display.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw graph for each set of values
            bar_height = 30
            bar_width = 300
            y_start = 100
            
            # Raw values on the left
            x_start_raw = 150
            
            # Smoothed values on the right
            x_start_smooth = 650
            
            # Add a vertical line to separate the two sections
            cv2.line(display, (width // 2, 50), (width // 2, height - 50), (100, 100, 100), 2)

            for i, emotion in enumerate(emotions):
                # Draw emotion label (centered)
                cv2.putText(display, emotion, (20, y_start + i*60 + bar_height//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # --- Raw values (left side) ---
                # Draw empty bar
                cv2.rectangle(display, (x_start_raw, y_start + i*60),
                              (x_start_raw + bar_width, y_start + i*60 + bar_height),
                              (50, 50, 50), -1)
                
                # Draw filled portion with raw value
                filled_width = int(bar_width * current_raw_values[i])
                cv2.rectangle(display, (x_start_raw, y_start + i*60),
                             (x_start_raw + filled_width, y_start + i*60 + bar_height),
                             emotion_colors[i], -1)
                
                # Add value text
                value_text = f"{current_raw_values[i]:.2f}"
                cv2.putText(display, value_text, 
                            (x_start_raw + bar_width + 10, y_start + i*60 + bar_height - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                # --- Smoothed values (right side) ---
                # Draw empty bar
                cv2.rectangle(display, (x_start_smooth, y_start + i*60),
                              (x_start_smooth + bar_width, y_start + i*60 + bar_height),
                              (50, 50, 50), -1)
                
                # Draw filled portion with smoothed value
                filled_width = int(bar_width * smoothed_probs[i])
                cv2.rectangle(display, (x_start_smooth, y_start + i*60),
                             (x_start_smooth + filled_width, y_start + i*60 + bar_height),
                             emotion_colors[i], -1)
                
                # Add value text
                value_text = f"{smoothed_probs[i]:.2f}"
                cv2.putText(display, value_text, 
                            (x_start_smooth + bar_width + 10, y_start + i*60 + bar_height - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Add explanation text at bottom
            explanation = "This demo shows how the moving average smoothing works:"
            cv2.putText(display, explanation, (120, height - 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                        
            explanation2 = "Left side shows raw values that change randomly, right side shows smoothed values"
            cv2.putText(display, explanation2, (120, height - 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                        
            explanation3 = f"Smoothing uses a moving average over the past {smoothing_window_size} frames"
            cv2.putText(display, explanation3, (120, height - 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

            # Resize display if needed
            if display.shape[1] != args.display_width or display.shape[0] != args.display_height:
                display = cv2.resize(display, (args.display_width, args.display_height))

            # Show window
            cv2.imshow('Moving Average Smoothing Demonstration', display)

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
        cv2.destroyAllWindows()
        print("Demo stopped.")

if __name__ == "__main__":
    main()
