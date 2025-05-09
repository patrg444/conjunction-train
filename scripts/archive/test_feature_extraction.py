#!/usr/bin/env python3
"""
Test script to verify that our fixed audio feature extraction produces 88 features.
This runs a single video through the new feature extraction pipeline and reports
the dimensionality of the extracted features.
"""

import os
import sys
import numpy as np
import glob
import logging

# Import the feature extraction function from our fixed script
from multimodal_preprocess_fixed import extract_audio_from_video, extract_audio_functionals

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_feature_extraction(video_path):
    """Test the full feature extraction pipeline on a single video."""
    logging.info(f"Testing feature extraction on: {video_path}")
    
    # Step 1: Extract audio from video
    audio_path = extract_audio_from_video(video_path)
    if audio_path is None:
        logging.error("Failed to extract audio from video")
        return
    
    logging.info(f"Extracted audio to: {audio_path}")
    
    # Step 2: Extract audio features using our fixed function
    audio_features, timestamps = extract_audio_functionals(audio_path)
    
    if audio_features is None:
        logging.error("Failed to extract audio features")
        return
    
    # Step 3: Report feature dimensions
    feature_dim = audio_features.shape[1]
    logging.info(f"✅ Extracted {feature_dim} audio features!")
    
    if feature_dim >= 87:
        logging.info(f"SUCCESS: The feature dimension is {feature_dim}!")
        logging.info("This indicates we're correctly extracting eGeMAPS functionals (87+ features)")
        logging.info("We've fixed the 26-dimension LLD issue!")
    elif feature_dim <= 30:
        logging.error(f"FAILED: Still extracting only {feature_dim} features - likely still using LLDs")
        logging.error("Expected around 88 features from eGeMAPS functionals")
    else:
        logging.warning(f"UNEXPECTED: Feature dimension is {feature_dim}, neither ~26 (LLDs) nor ~88 (functionals)")
    
    # Print some sample feature values
    logging.info(f"Sample feature vector: {audio_features[0, :5]} ...")
    
    return feature_dim

def main():
    """Find a sample video and run the test."""
    # Look for test videos in common locations
    search_locations = [
        "data/test_videos",
        "data/sample_videos",
        "data",
        "."
    ]
    
    video_extensions = [".mp4", ".avi", ".mov", ".flv"]
    
    # Try to find a video file
    sample_video = None
    for location in search_locations:
        if os.path.exists(location):
            for ext in video_extensions:
                pattern = os.path.join(location, f"*{ext}")
                videos = glob.glob(pattern)
                if videos:
                    sample_video = videos[0]
                    break
            if sample_video:
                break
    
    if sample_video:
        logging.info(f"Found sample video: {sample_video}")
        feature_dim = test_feature_extraction(sample_video)
        if feature_dim >= 87:
            print("\n✅ SUCCESS: Audio feature extraction is fixed - producing {feature_dim} features!")
            print("This is a significant improvement from the original 26 LLD features.")
            return 0
        else:
            print(f"\n❌ FAILED: Audio feature extraction is not fixed! Got {feature_dim} features.")
            return 1
    else:
        logging.error("Could not find any video files to test with")
        print("Please specify a video file path as a command line argument")
        return 1

if __name__ == "__main__":
    # If a video path is provided as an argument, use that
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        if os.path.exists(video_path):
            test_feature_extraction(video_path)
        else:
            logging.error(f"Video file not found: {video_path}")
    else:
        main()
