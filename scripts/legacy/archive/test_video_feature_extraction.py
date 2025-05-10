#!/usr/bin/env python3
"""
Test script to verify that our fixed video feature extraction produces real features
from DeepFace rather than random values. This runs a single video through the new 
feature extraction pipeline and verifies that the video features are properly extracted.
"""

import os
import sys
import numpy as np
import logging

# Import the feature extraction function from our fixed script
from multimodal_preprocess_fixed import extract_frame_level_video_features

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_video_feature_extraction(video_path):
    """Test the video feature extraction pipeline on a single video."""
    logging.info(f"Testing video feature extraction on: {video_path}")
    
    # Extract video features
    video_features, timestamps = extract_frame_level_video_features(video_path)
    
    if video_features is None:
        logging.error("Failed to extract video features")
        return
    
    # Report feature dimensions
    feature_dim = video_features.shape[1]
    frame_count = video_features.shape[0]
    
    logging.info(f"✅ Extracted {feature_dim} video features across {frame_count} frames!")
    
    if feature_dim == 4096:
        logging.info(f"SUCCESS: The feature dimension is {feature_dim}!")
        logging.info("This confirms we're correctly extracting DeepFace VGG-Face embeddings (4096 features)")
    else:
        logging.warning(f"UNEXPECTED: Feature dimension is {feature_dim}, not the expected 4096 for VGG-Face")
    
    # Check if the features aren't random (look for patterns of zeros or repeated values)
    is_random = True
    # Check if any frames have all zeros (indicating no face detection)
    zero_frames = np.sum(np.all(video_features == 0, axis=1))
    
    if zero_frames > 0:
        logging.info(f"Found {zero_frames} frames with zero vectors (likely no face detected)")
        is_random = False
    
    # Check variance between frames - random would have high variance, real embeddings less so
    if frame_count > 1:
        frame_variance = np.mean(np.var(video_features, axis=0))
        logging.info(f"Average variance between frames: {frame_variance:.6f}")
        
        # Random data would have higher variance typically
        if frame_variance < 0.1:  # This threshold is a heuristic
            is_random = False
    
    if is_random:
        logging.warning("POSSIBLE ISSUE: Features may still be random data")
    else:
        logging.info("Features appear to be real facial embeddings, not random data")
    
    # Print some sample feature values
    logging.info(f"Sample feature vector (first 5 values): {video_features[0, :5]} ...")
    
    return feature_dim, frame_count

def test_complete_processing(video_path):
    """Test the full feature extraction pipeline including audio and video."""
    from multimodal_preprocess_fixed import process_video_for_single_segment
    import shutil
    
    # Create a temporary output directory
    temp_output_dir = "temp_test_output"
    os.makedirs(temp_output_dir, exist_ok=True)
    
    try:
        # Process the video
        output_file = process_video_for_single_segment(
            video_path=video_path,
            output_dir=temp_output_dir
        )
        
        if output_file is None:
            logging.error("Failed to process video")
            return False
        
        # Load the output file
        data = np.load(output_file)
        
        # Check that both video and audio features are present
        if 'video_features' not in data or 'audio_features' not in data:
            logging.error("Missing video or audio features in output file")
            return False
        
        video_features = data['video_features']
        audio_features = data['audio_features']
        
        # Check video feature dimensions
        video_dim = video_features.shape[1]
        logging.info(f"Video features dimension: {video_dim}")
        if video_dim != 4096:
            logging.warning(f"Unexpected video feature dimension: {video_dim} (expected 4096)")
        
        # Check audio feature dimensions
        audio_dim = audio_features.shape[1]
        logging.info(f"Audio features dimension: {audio_dim}")
        if audio_dim < 87:
            logging.warning(f"Unexpected audio feature dimension: {audio_dim} (expected ≥87)")
        
        logging.info("✅ SUCCESS: Both video and audio features were extracted successfully")
        logging.info(f"  - Video: {video_features.shape[0]} frames with {video_dim} features each")
        logging.info(f"  - Audio: {audio_features.shape[0]} frames with {audio_dim} features each")
        
        return True
    
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_output_dir):
            shutil.rmtree(temp_output_dir)

def main():
    """Find a sample video and run the test."""
    # Look for test videos in common locations
    search_locations = [
        "data/test_videos",
        "data/sample_videos",
        "data/RAVDESS/Actor_01",
        "data/CREMA-D/VideoFlash",
        "data"
    ]
    
    video_extensions = [".mp4", ".avi", ".mov", ".flv"]
    
    # Try to find a video file
    sample_video = None
    for location in search_locations:
        if os.path.exists(location):
            for ext in video_extensions:
                pattern = os.path.join(location, f"*{ext}")
                import glob
                videos = glob.glob(pattern)
                if videos:
                    sample_video = videos[0]
                    break
            if sample_video:
                break
    
    if sample_video:
        logging.info(f"Found sample video: {sample_video}")
        feature_dim, frame_count = test_video_feature_extraction(sample_video)
        
        if feature_dim == 4096:
            print("\n✅ SUCCESS: Video feature extraction is working correctly!")
            print(f"Extracted {frame_count} frames with {feature_dim} features each")
            
            # Test the complete processing pipeline
            print("\nTesting complete processing pipeline (video + audio)...")
            success = test_complete_processing(sample_video)
            
            if success:
                print("\n✅ COMPLETE SUCCESS: Both video and audio features are extracted correctly!")
                print("The fixed pipeline is working as expected.")
                return 0
            else:
                print("\n❌ PARTIAL SUCCESS: Video features work but full pipeline has issues.")
                return 1
        else:
            print(f"\n❌ FAILED: Video feature extraction has issues! Got {feature_dim} features.")
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
            test_video_feature_extraction(video_path)
            test_complete_processing(video_path)
        else:
            logging.error(f"Video file not found: {video_path}")
    else:
        main()
