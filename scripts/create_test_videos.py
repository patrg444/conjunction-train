#!/usr/bin/env python3
"""
Script to create test videos for testing the video feature extraction functionality.
This creates a video with a solid color background and a moving circle to simulate a basic
test case (although this will likely result in no face detections).
"""

import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

def create_simple_test_video(output_path, duration=5, fps=30, width=640, height=480):
    """Create a simple test video with a moving circle."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frames = int(duration * fps)
    radius = 50
    
    for i in tqdm(range(frames), desc="Creating frames"):
        # Create a frame with gray background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 200
        
        # Calculate position of circle (moving in a circle)
        t = i / frames * 2 * np.pi  # Time parameter from 0 to 2π
        center_x = int(width/2 + np.cos(t) * (width/4))
        center_y = int(height/2 + np.sin(t) * (height/4))
        
        # Draw circle
        cv2.circle(frame, (center_x, center_y), radius, (0, 0, 255), -1)
        
        # Add frame count text
        cv2.putText(frame, f"Frame: {i}/{frames}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Write frame to video
        out.write(frame)
    
    out.release()
    print(f"Created test video: {output_path}")
    return output_path

def use_webcam_to_create_test_video(output_path, duration=5, fps=30):
    """Create a test video using the webcam (if available)."""
    # Try to open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Falling back to synthetic video.")
        return create_simple_test_video(output_path, duration, fps)
    
    # Get webcam properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frames = int(duration * fps)
    frame_count = 0
    
    print("Recording from webcam. Press 'q' to stop early.")
    pbar = tqdm(total=frames, desc="Recording")
    
    while frame_count < frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Add frame count text
        cv2.putText(frame, f"Frame: {frame_count}/{frames}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Recording Test Video', frame)
        
        # Write frame to video
        out.write(frame)
        
        frame_count += 1
        pbar.update(1)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    pbar.close()
    
    print(f"Created test video from webcam: {output_path}")
    return output_path

def create_sample_dataset(output_dir="test_videos", num_videos=3, duration=2):
    """Create a small sample dataset of test videos."""
    os.makedirs(output_dir, exist_ok=True)
    
    video_paths = []
    for i in range(num_videos):
        video_path = os.path.join(output_dir, f"test_video_{i+1}.mp4")
        create_simple_test_video(video_path, duration=duration)
        video_paths.append(video_path)
    
    print(f"Created {num_videos} test videos in {output_dir}")
    return video_paths

def main():
    """Main function for creating test videos."""
    parser = argparse.ArgumentParser(description="Create test videos for feature extraction testing")
    parser.add_argument("--output", default="test_video.mp4", help="Output video path")
    parser.add_argument("--duration", type=int, default=5, help="Video duration in seconds")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--webcam", action="store_true", help="Use webcam to create video (if available)")
    parser.add_argument("--dataset", action="store_true", help="Create a small sample dataset")
    
    args = parser.parse_args()
    
    if args.dataset:
        create_sample_dataset()
    elif args.webcam:
        use_webcam_to_create_test_video(args.output, args.duration, args.fps)
    else:
        create_simple_test_video(args.output, args.duration, args.fps)
    
    print("Video creation completed.")
    return 0

if __name__ == "__main__":
    main()
