#!/usr/bin/env python3
"""Test script to verify the FPS detection and resampling avoidance in multimodal_preprocess_fixed."""

import os
import logging
import sys
from multimodal_preprocess_fixed import resample_video

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # Log to console for this test
    ]
)

def test_fps_detection(video_path):
    """Test the FPS detection and resampling logic."""
    logging.info(f"Testing FPS detection on: {video_path}")
    
    # Run the resample_video function with the video
    output_path = resample_video(video_path, fps=15)
    
    # Check if the output path is the same as the input path (indicating no resampling was done)
    if os.path.abspath(output_path) == os.path.abspath(video_path):
        logging.info("SUCCESS: Video was detected as already having correct fps, no resampling needed!")
    else:
        logging.info(f"Video was resampled to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_fps_detection(sys.argv[1])
    else:
        # Use a default video from downsampled_videos
        test_fps_detection("downsampled_videos/RAVDESS/Actor_01/01-01-01-01-01-01-01.mp4")
