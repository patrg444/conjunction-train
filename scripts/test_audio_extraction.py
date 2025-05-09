#!/usr/bin/env python3
"""Test script to verify audio extraction from the downsampled RAVDESS videos."""

import os
import sys
import logging
from multimodal_preprocess_fixed import extract_audio_from_video

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # Log to console for this test
    ]
)

def test_audio_extraction(video_path, output_dir="temp_extracted_audio"):
    """Test the audio extraction from a video file."""
    logging.info(f"Testing audio extraction from: {video_path}")
    
    # Extract audio using the function from multimodal_preprocess_fixed
    audio_path = extract_audio_from_video(video_path, output_dir)
    
    if audio_path and os.path.exists(audio_path):
        audio_size = os.path.getsize(audio_path) / 1024  # Size in KB
        logging.info(f"SUCCESS: Audio successfully extracted to: {audio_path} (Size: {audio_size:.2f} KB)")
        return True
    else:
        logging.error(f"FAILED: Audio extraction failed for: {video_path}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_audio_extraction(sys.argv[1])
    else:
        # Use a default video from downsampled_videos
        test_audio_extraction("downsampled_videos/RAVDESS/Actor_01/01-01-01-01-01-01-01.mp4")
