#!/usr/bin/env python3
"""
Script to check and compare properties of specific original and downsampled videos,
and analyze the extracted features.
"""

import os
import sys
import numpy as np
from moviepy.editor import VideoFileClip
import cv2

def get_video_info(video_path):
    """Get detailed information about a video file using both moviepy and OpenCV."""
    print(f"\nAnalyzing: {video_path}")
    print("-" * 60)
    
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"File doesn't exist: {video_path}")
        return None
        
    try:
        # Get info using moviepy
        video = VideoFileClip(video_path)
        moviepy_duration = video.duration
        moviepy_fps = video.fps
        moviepy_frame_count = int(moviepy_duration * moviepy_fps)
        
        print("Using MoviePy:")
        print(f"  Duration: {moviepy_duration:.2f} seconds")
        print(f"  FPS: {moviepy_fps:.2f}")
        print(f"  Estimated frame count: {moviepy_frame_count}")
        
        # Get info using OpenCV
        cap = cv2.VideoCapture(video_path)
        opencv_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        opencv_fps = cap.get(cv2.CAP_PROP_FPS)
        opencv_duration = opencv_frame_count / opencv_fps if opencv_fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print("\nUsing OpenCV:")
        print(f"  Duration: {opencv_duration:.2f} seconds")
        print(f"  FPS: {opencv_fps:.2f}")
        print(f"  Frame count: {opencv_frame_count}")
        print(f"  Resolution: {width}x{height}")
        
        # Count frames manually with OpenCV for verification
        manual_frame_count = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            manual_frame_count += 1
        
        print(f"  Actual frame count (manually counted): {manual_frame_count}")
        
        # Close resources
        video.close()
        cap.release()
        
        return {
            "path": video_path,
            "duration_moviepy": moviepy_duration,
            "fps_moviepy": moviepy_fps,
            "frame_count_moviepy": moviepy_frame_count,
            "duration_opencv": opencv_duration,
            "fps_opencv": opencv_fps,
            "frame_count_opencv": opencv_frame_count,
            "manual_frame_count": manual_frame_count,
            "resolution": f"{width}x{height}"
        }
    
    except Exception as e:
        print(f"Error analyzing video: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

def analyze_npz_file(npz_path):
    """Analyze the NPZ file with extracted features."""
    print(f"\nAnalyzing extracted features: {npz_path}")
    print("-" * 60)
    
    if not os.path.exists(npz_path):
        print(f"File doesn't exist: {npz_path}")
        return
    
    try:
        # Load the NPZ file
        data = np.load(npz_path)
        
        # Extract and print information
        print("Feature file contents:")
        for key in data.files:
            if isinstance(data[key], np.ndarray):
                print(f"  {key}: shape {data[key].shape}, type {data[key].dtype}")
            else:
                print(f"  {key}: {data[key]}")
        
        # Calculate durations based on features and timestamps
        if 'video_features' in data and 'video_timestamps' in data:
            video_frames = data['video_features'].shape[0]
            if video_frames > 1 and len(data['video_timestamps']) > 1:
                # Calculate implied fps from timestamps
                video_duration = data['video_timestamps'][-1] - data['video_timestamps'][0]
                implied_fps = (video_frames - 1) / video_duration if video_duration > 0 else 0
                print(f"\nVideo feature analysis:")
                print(f"  Video frames: {video_frames}")
                print(f"  Video duration (from timestamps): {video_duration:.2f} seconds")
                print(f"  Implied video FPS: {implied_fps:.2f}")
                print(f"  Feature dimensions per frame: {data['video_features'].shape[1]}")
                
                if implied_fps > 0:
                    # Also calculate duration assuming evenly spaced frames
                    calculated_duration = video_frames / implied_fps
                    print(f"  Calculated duration (frames/fps): {calculated_duration:.2f} seconds")
        
        if 'audio_features' in data and 'audio_timestamps' in data:
            audio_frames = data['audio_features'].shape[0]
            if audio_frames > 1 and len(data['audio_timestamps']) > 1:
                # Calculate implied fps from timestamps
                audio_duration = data['audio_timestamps'][-1] - data['audio_timestamps'][0]
                implied_fps = (audio_frames - 1) / audio_duration if audio_duration > 0 else 0
                print(f"\nAudio feature analysis:")
                print(f"  Audio frames: {audio_frames}")
                print(f"  Audio duration (from timestamps): {audio_duration:.2f} seconds")
                print(f"  Implied audio sampling rate: {implied_fps:.2f} Hz")
                print(f"  Feature dimensions per frame: {data['audio_features'].shape[1]}")
    
    except Exception as e:
        print(f"Error analyzing NPZ file: {str(e)}")
        import traceback
        print(traceback.format_exc())

def main():
    # Define the paths to analyze
    original_video = "data/CREMA-D/1001_DFA_ANG_XX.flv"
    downsampled_video = "downsampled_videos/CREMA-D-audio-complete/1001_DFA_ANG_XX.flv"
    features_file = "test_single_file_output/1001_DFA_ANG_XX.npz"
    
    # Check if the resampled video exists in the temp directory
    resampled_video = "temp_resampled_videos/resampled_15fps_1001_DFA_ANG_XX.flv"
    
    # Analyze the original video
    original_info = get_video_info(original_video)
    
    # Analyze the downsampled video
    downsampled_info = get_video_info(downsampled_video)
    
    # Analyze the resampled (15 fps) video if it exists
    if os.path.exists(resampled_video):
        resampled_info = get_video_info(resampled_video)
    else:
        print(f"\nResampled video not found: {resampled_video}")
        # Try to find it elsewhere
        possible_paths = [
            "test_single_file_output/temp_resampled_videos/resampled_15fps_1001_DFA_ANG_XX.flv",
            "downsampled_videos/CREMA-D/temp_resampled_videos/resampled_15fps_1001_DFA_ANG_XX.flv"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Found resampled video at: {path}")
                resampled_info = get_video_info(path)
                break
    
    # Analyze the extracted features
    analyze_npz_file(features_file)
    
    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    if original_info and downsampled_info:
        original_fps = original_info["fps_opencv"]
        downsampled_fps = downsampled_info["fps_opencv"]
        
        fps_ratio = downsampled_fps / original_fps if original_fps > 0 else 0
        print(f"FPS ratio (downsampled/original): {fps_ratio:.2f}")
        
        if original_info["frame_count_opencv"] > 0:
            frame_ratio = downsampled_info["frame_count_opencv"] / original_info["frame_count_opencv"]
            print(f"Frame count ratio (downsampled/original): {frame_ratio:.2f}")
        
        duration_ratio = downsampled_info["duration_opencv"] / original_info["duration_opencv"] if original_info["duration_opencv"] > 0 else 0
        print(f"Duration ratio (downsampled/original): {duration_ratio:.2f}")
        
        # Calculate expected frame count at 15 fps
        original_duration = original_info["duration_opencv"]
        expected_frames_at_15fps = int(original_duration * 15)
        print(f"\nIf the original video ({original_duration:.2f} seconds) is downsampled to 15 fps:")
        print(f"Expected frame count: {expected_frames_at_15fps}")
        
        # Video information
        print("\nOriginal video:")
        print(f"  Duration: {original_info['duration_opencv']:.2f} seconds")
        print(f"  FPS: {original_info['fps_opencv']:.2f}")
        print(f"  Frame count: {original_info['frame_count_opencv']}")
        
        print("\nDownsampled video:")
        print(f"  Duration: {downsampled_info['duration_opencv']:.2f} seconds")
        print(f"  FPS: {downsampled_info['fps_opencv']:.2f}")
        print(f"  Frame count: {downsampled_info['frame_count_opencv']}")
        
        if 'resampled_info' in locals():
            print("\nResampled video (15 fps):")
            print(f"  Duration: {resampled_info['duration_opencv']:.2f} seconds")
            print(f"  FPS: {resampled_info['fps_opencv']:.2f}")
            print(f"  Frame count: {resampled_info['frame_count_opencv']}")
            print(f"  Detected frame count closely matches expected count: {'Yes' if abs(resampled_info['frame_count_opencv'] - expected_frames_at_15fps) <= 1 else 'No'}")
    
    # Verification of downsampling math
    print("\nVerification of 35 frames at 15 fps:")
    print(f"Duration from frame count: 35 frames ÷ 15 fps = {35/15:.2f} seconds")
    print(f"Does this match the expected content duration? Compare with original duration: {original_info['duration_opencv']:.2f} seconds")
    print(f"Approximate coverage percentage: {(35/15) / original_info['duration_opencv'] * 100:.1f}% of original content")

if __name__ == "__main__":
    main()
