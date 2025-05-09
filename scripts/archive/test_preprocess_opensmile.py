#!/usr/bin/env python3
"""
Test script to verify the openSMILE integration in the preprocess.py module.
This script tests the audio feature extraction part specifically.
"""

import os
import logging
import numpy as np
from moviepy.editor import VideoFileClip
from preprocess import extract_audio_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("test_preprocess_opensmile.log"),
        logging.StreamHandler()
    ]
)

def main():
    print("Testing openSMILE integration with correct command-line arguments...")
    
    # Use our test videos
    test_videos_dir = "test_videos"
    first_test_video = os.path.join(test_videos_dir, "1001_TEST_ANG_XX.mp4")
    
    # Extract the audio to a temporary directory
    temp_dir = "temp_test_extract"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Extract audio using moviepy
    video = VideoFileClip(first_test_video)
    audio = video.audio
    audio_path = os.path.join(temp_dir, "test_audio.wav")
    audio.write_audiofile(audio_path, verbose=False, logger=None)
    video.close()
    
    print(f"Extracted test audio to: {audio_path}")
    
    # Config file path that works with the openSMILE installed on this system
    config_file = "opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/config/egemaps/v02/eGeMAPSv02.conf"
    
    # Run the audio feature extraction with our updated implementation
    print(f"Extracting audio features with openSMILE...")
    features = extract_audio_features([audio_path], config_file, output_dir=temp_dir)
    
    if features.size > 0:
        print(f"✅ Successfully extracted audio features using openSMILE!")
        print(f"Extracted features shape: {features.shape}")
        print(f"Feature dimension: {features.shape[1]}")
        return True
    else:
        print(f"❌ Failed to extract audio features using openSMILE!")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        import sys
        sys.exit(1)
