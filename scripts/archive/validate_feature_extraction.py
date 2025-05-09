#!/usr/bin/env python3
"""
This script validates that both video and audio features are correctly extracted
in the processed RAVDESS and CREMA-D files. It examines a sample of processed files
to ensure they contain the proper dimensionality for both features.
"""

import os
import numpy as np
import glob
import logging
from tqdm import tqdm
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def validate_processed_files(directory, sample_size=10):
    """
    Validate a sample of processed .npz files to ensure they contain
    properly extracted video and audio features.
    
    Args:
        directory: Directory containing processed .npz files
        sample_size: Number of files to sample for validation
    
    Returns:
        True if validation passes, False otherwise
    """
    logging.info(f"Validating processed files in: {directory}")
    
    # Get all npz files
    npz_files = glob.glob(os.path.join(directory, "*.npz"))
    if not npz_files:
        logging.error(f"No .npz files found in {directory}")
        return False
    
    # Sample files
    import random
    if len(npz_files) > sample_size:
        sample_files = random.sample(npz_files, sample_size)
    else:
        sample_files = npz_files
    
    logging.info(f"Examining {len(sample_files)} files out of {len(npz_files)} total")
    
    # Track validation results
    video_dims = []
    audio_dims = []
    missing_features = []
    
    # Expected dimensions
    expected_video_dim = 4096  # VGG-Face embeddings
    expected_min_audio_dim = 87  # eGeMAPS functionals (at least 87 features)
    
    # Check each file
    for npz_file in tqdm(sample_files, desc="Validating files"):
        try:
            data = np.load(npz_file)
            
            # Check if both feature types exist
            has_video = 'video_features' in data
            has_audio = 'audio_features' in data
            
            if not has_video or not has_audio:
                missing_features.append(npz_file)
                if not has_video:
                    logging.warning(f"Missing video features in: {os.path.basename(npz_file)}")
                if not has_audio:
                    logging.warning(f"Missing audio features in: {os.path.basename(npz_file)}")
                continue
            
            # Check dimensions
            video_feature_dim = data['video_features'].shape[1]
            audio_feature_dim = data['audio_features'].shape[1]
            
            video_dims.append(video_feature_dim)
            audio_dims.append(audio_feature_dim)
            
        except Exception as e:
            logging.error(f"Error processing {npz_file}: {str(e)}")
    
    # Calculate statistics
    if video_dims:
        avg_video_dim = sum(video_dims) / len(video_dims)
        min_video_dim = min(video_dims)
        max_video_dim = max(video_dims)
        
        logging.info(f"Video feature dimensions: min={min_video_dim}, max={max_video_dim}, avg={avg_video_dim:.2f}")
        
        if min_video_dim != expected_video_dim or max_video_dim != expected_video_dim:
            logging.warning(f"Video feature dimensions vary! Expected {expected_video_dim} consistently")
    else:
        logging.error("Could not analyze video dimensions - no valid files found")
    
    if audio_dims:
        avg_audio_dim = sum(audio_dims) / len(audio_dims)
        min_audio_dim = min(audio_dims)
        max_audio_dim = max(audio_dims)
        
        logging.info(f"Audio feature dimensions: min={min_audio_dim}, max={max_audio_dim}, avg={avg_audio_dim:.2f}")
        
        if min_audio_dim < expected_min_audio_dim:
            logging.warning(f"Some audio features have less than {expected_min_audio_dim} dimensions!")
    else:
        logging.error("Could not analyze audio dimensions - no valid files found")
    
    # Print summary
    print("\n--- Validation Summary ---")
    print(f"Directory: {directory}")
    print(f"Files checked: {len(sample_files)} of {len(npz_files)} total")
    print(f"Files missing features: {len(missing_features)}")
    
    if video_dims:
        print(f"Video features: {min_video_dim}-{max_video_dim} dimensions (expected {expected_video_dim})")
        if min_video_dim == expected_video_dim and max_video_dim == expected_video_dim:
            print("✅ Video feature extraction is CORRECT")
        else:
            print("❌ Video feature extraction has ISSUES")
    
    if audio_dims:
        print(f"Audio features: {min_audio_dim}-{max_audio_dim} dimensions (expected ≥{expected_min_audio_dim})")
        if min_audio_dim >= expected_min_audio_dim:
            print("✅ Audio feature extraction is CORRECT")
        else:
            print("❌ Audio feature extraction has ISSUES")
    
    # Overall validation result
    if (video_dims and min_video_dim == expected_video_dim and max_video_dim == expected_video_dim and
        audio_dims and min_audio_dim >= expected_min_audio_dim and not missing_features):
        print("\n✅ VALIDATION PASSED: Both video and audio features look correct!")
        return True
    else:
        print("\n❌ VALIDATION FAILED: Issues detected with feature extraction")
        return False

def main():
    # Default directories
    ravdess_dir = "processed_ravdess_fixed"
    cremad_dir = "processed_crema_d_fixed"
    
    # Check if custom directories provided
    if len(sys.argv) > 2:
        ravdess_dir = sys.argv[1]
        cremad_dir = sys.argv[2]
    
    # Validate both directories
    ravdess_passed = False
    cremad_passed = False
    
    if os.path.exists(ravdess_dir) and os.listdir(ravdess_dir):
        print("\n=== Validating RAVDESS Dataset ===")
        ravdess_passed = validate_processed_files(ravdess_dir)
    else:
        print(f"\n⚠️ Cannot validate RAVDESS: Directory {ravdess_dir} does not exist or is empty")
    
    if os.path.exists(cremad_dir) and os.listdir(cremad_dir):
        print("\n=== Validating CREMA-D Dataset ===")
        cremad_passed = validate_processed_files(cremad_dir)
    else:
        print(f"\n⚠️ Cannot validate CREMA-D: Directory {cremad_dir} does not exist or is empty")
    
    # Final result
    if ravdess_passed and cremad_passed:
        print("\n✅ VALIDATION COMPLETE: Both datasets have correct video and audio features!")
        return 0
    elif (ravdess_passed and not os.path.exists(cremad_dir)) or (cremad_passed and not os.path.exists(ravdess_dir)):
        print("\n⚠️ PARTIAL VALIDATION: Available dataset(s) passed validation")
        return 0
    else:
        print("\n❌ VALIDATION FAILED: Issues detected in one or both datasets")
        return 1

if __name__ == "__main__":
    sys.exit(main())
