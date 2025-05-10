#!/usr/bin/env python3
"""
Create synthetic test videos for testing the multimodal preprocessing pipeline.
This script generates simple videos with moving shapes and sounds that simulate
emotional expressions to test our synchronization implementation.
"""

import os
import numpy as np
from moviepy.editor import VideoClip, AudioClip, CompositeVideoClip
import argparse
from tqdm import tqdm

def make_emotion_video_frame(t, emotion, width, height, duration):
    """Create a video frame simulating an emotion at time t."""
    # Create a black background
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Emotion parameters (color, movement speed, size)
    emotion_params = {
        "ANG": {"color": (255, 0, 0), "speed": 1.5, "size": 1.2},  # Red, fast, larger
        "DIS": {"color": (0, 255, 0), "speed": 0.8, "size": 0.9},  # Green, slow, smaller
        "FEA": {"color": (128, 0, 255), "speed": 2.0, "size": 0.8},  # Purple, very fast, smaller
        "HAP": {"color": (255, 255, 0), "speed": 1.2, "size": 1.1},  # Yellow, moderate, larger
        "NEU": {"color": (128, 128, 128), "speed": 0.7, "size": 1.0},  # Gray, slowish, normal
        "SAD": {"color": (0, 0, 255), "speed": 0.5, "size": 0.7}  # Blue, slow, small
    }
    
    params = emotion_params.get(emotion, emotion_params["NEU"])
    
    # Calculate position of the shape based on time and emotion
    speed = params["speed"]
    x = int(width * 0.5 + width * 0.3 * np.sin(t * speed * np.pi))
    y = int(height * 0.5 + height * 0.3 * np.cos(t * speed * np.pi))
    
    # Draw a colored oval/circle (face)
    size = int(min(width, height) * 0.25 * params["size"])
    color = params["color"]
    
    # Draw the oval face
    for i in range(height):
        for j in range(width):
            # Calculate distance to center
            dx = (j - x) / (size * 1.2)  # Slightly wider
            dy = (i - y) / size
            dist = dx*dx + dy*dy
            
            if dist < 1.0:
                img[i, j] = color
    
    # Add eyes (two small black circles)
    eye_size = size // 5
    eye_offset_x = size // 3
    eye_offset_y = size // 8
    
    # Left eye
    for i in range(max(0, y-eye_offset_y-eye_size), min(height, y-eye_offset_y+eye_size)):
        for j in range(max(0, x-eye_offset_x-eye_size), min(width, x-eye_offset_x+eye_size)):
            dx = (j - (x-eye_offset_x)) / eye_size
            dy = (i - (y-eye_offset_y)) / eye_size
            dist = dx*dx + dy*dy
            if dist < 0.7:
                img[i, j] = (0, 0, 0)  # Black
    
    # Right eye
    for i in range(max(0, y-eye_offset_y-eye_size), min(height, y-eye_offset_y+eye_size)):
        for j in range(max(0, x+eye_offset_x-eye_size), min(width, x+eye_offset_x+eye_size)):
            dx = (j - (x+eye_offset_x)) / eye_size
            dy = (i - (y-eye_offset_y)) / eye_size
            dist = dx*dx + dy*dy
            if dist < 0.7:
                img[i, j] = (0, 0, 0)  # Black
    
    # Add a mouth that varies by emotion
    mouth_width = size // 2
    mouth_height = size // 10
    mouth_y_offset = size // 3
    
    # Mouth shape varies by emotion
    if emotion == "HAP":  # Happy - smile
        for i in range(max(0, y+mouth_y_offset-mouth_height), min(height, y+mouth_y_offset+mouth_height)):
            for j in range(max(0, x-mouth_width), min(width, x+mouth_width)):
                dx = (j - x) / mouth_width
                dy = (i - (y+mouth_y_offset)) / mouth_height - 1.0 * dx * dx
                if abs(dy) < 0.5:
                    img[i, j] = (0, 0, 0)  # Black
    
    elif emotion == "SAD":  # Sad - frown
        for i in range(max(0, y+mouth_y_offset-mouth_height), min(height, y+mouth_y_offset+mouth_height)):
            for j in range(max(0, x-mouth_width), min(width, x+mouth_width)):
                dx = (j - x) / mouth_width
                dy = (i - (y+mouth_y_offset)) / mouth_height + 1.0 * dx * dx
                if abs(dy) < 0.5:
                    img[i, j] = (0, 0, 0)  # Black
    
    elif emotion == "ANG":  # Angry - straight mouth with furrowed brow
        for i in range(max(0, y+mouth_y_offset-mouth_height), min(height, y+mouth_y_offset+mouth_height)):
            for j in range(max(0, x-mouth_width), min(width, x+mouth_width)):
                if abs(i - (y+mouth_y_offset)) < mouth_height:
                    img[i, j] = (0, 0, 0)  # Black
                    
        # Add angry eyebrows
        eyebrow_width = eye_size * 2
        eyebrow_height = eye_size // 2
        eyebrow_y_offset = eye_offset_y * 2
        
        # Left eyebrow
        for i in range(max(0, y-eyebrow_y_offset-eyebrow_height), min(height, y-eyebrow_y_offset+eyebrow_height)):
            for j in range(max(0, x-eye_offset_x-eyebrow_width), min(width, x-eye_offset_x+eyebrow_width)):
                dx = (j - (x-eye_offset_x)) / eyebrow_width
                dy = (i - (y-eyebrow_y_offset)) / eyebrow_height - 0.5 * dx
                if abs(dy) < 0.5:
                    img[i, j] = (0, 0, 0)  # Black
        
        # Right eyebrow
        for i in range(max(0, y-eyebrow_y_offset-eyebrow_height), min(height, y-eyebrow_y_offset+eyebrow_height)):
            for j in range(max(0, x+eye_offset_x-eyebrow_width), min(width, x+eye_offset_x+eyebrow_width)):
                dx = (j - (x+eye_offset_x)) / eyebrow_width
                dy = (i - (y-eyebrow_y_offset)) / eyebrow_height + 0.5 * dx
                if abs(dy) < 0.5:
                    img[i, j] = (0, 0, 0)  # Black
    
    else:  # Neutral/other - straight mouth
        for i in range(max(0, y+mouth_y_offset-mouth_height), min(height, y+mouth_y_offset+mouth_height)):
            for j in range(max(0, x-mouth_width), min(width, x+mouth_width)):
                if abs(i - (y+mouth_y_offset)) < mouth_height:
                    img[i, j] = (0, 0, 0)  # Black
    
    return img

def make_emotion_audio(t, emotion, duration):
    """Create an audio sample simulating an emotion at time t."""
    # Emotion parameters (base frequency, amplitude, modulation)
    emotion_params = {
        "ANG": {"freq": 300, "amp": 0.8, "mod": 2.0},  # Higher pitch, loud, more variation
        "DIS": {"freq": 200, "amp": 0.5, "mod": 1.5},  # Lower, medium volume, some variation
        "FEA": {"freq": 350, "amp": 0.6, "mod": 3.0},  # Higher, medium-loud, lots of variation
        "HAP": {"freq": 250, "amp": 0.7, "mod": 1.2},  # Medium pitch, medium-loud, some variation
        "NEU": {"freq": 220, "amp": 0.4, "mod": 0.5},  # Medium-low, quieter, minimal variation
        "SAD": {"freq": 180, "amp": 0.3, "mod": 0.7}   # Low, quiet, slow variation
    }
    
    params = emotion_params.get(emotion, emotion_params["NEU"])
    
    # Base frequency with emotion-based modulation
    frequency = params["freq"] * (1 + 0.1 * params["mod"] * np.sin(t * 2 * np.pi / duration))
    amplitude = params["amp"] * (1 + 0.1 * params["mod"] * np.sin(t * 5 * np.pi / duration))
    
    # Create the tone
    return amplitude * np.sin(2 * np.pi * frequency * t)

def create_emotion_video(output_path, emotion="NEU", duration=3.0, fps=30):
    """Create a test video simulating an emotion."""
    width, height = 320, 240
    
    # Create the video clip
    def make_frame(t):
        return make_emotion_video_frame(t, emotion, width, height, duration)
    
    video = VideoClip(make_frame, duration=duration)
    
    # Create the audio clip
    def make_audio(t):
        return make_emotion_audio(t, emotion, duration)
    
    audio = AudioClip(make_audio, duration=duration, fps=44100)
    
    # Combine video and audio
    video = video.set_audio(audio)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Write the video file
    video.write_videofile(output_path, fps=fps, codec="libx264", audio_codec="aac")
    
    return output_path

def create_test_dataset(output_dir="test_videos", duration=3.0, fps=30):
    """Create a small test dataset of emotional videos."""
    os.makedirs(output_dir, exist_ok=True)
    
    emotions = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
    
    # Create a few test videos for each emotion
    for emotion in tqdm(emotions, desc="Creating test videos"):
        # Create multiple instances for each emotion (like in CREMA-D)
        for i in range(1, 4):  # Create 3 samples per emotion
            output_path = os.path.join(output_dir, f"{1000+i}_TEST_{emotion}_XX.mp4")
            create_emotion_video(output_path, emotion, duration, fps)
    
    print(f"Created test dataset in {output_dir}")
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create test videos for multimodal processing")
    parser.add_argument("--output-dir", type=str, default="test_videos", 
                        help="Directory to save test videos")
    parser.add_argument("--duration", type=float, default=3.0,
                        help="Duration of each video in seconds")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second for the videos")
    
    args = parser.parse_args()
    
    create_test_dataset(args.output_dir, args.duration, args.fps)
