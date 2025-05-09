#!/usr/bin/env python3
"""
Run the fixed emotion recognition pipeline with both critical issues resolved:

1. Audio Feature Dimension Issue:
   - Using full 88-dimension eGeMAPS functionals instead of 26-dimension LLDs
   - Properly extracts features using csvoutput instead of lldcsvoutput

2. 7-Class vs 6-Class Model Issue:
   - Explicitly excludes the "Surprised" class from RAVDESS
   - Configures a 6-class model instead of a 7-class model

3. Memory Issue:
   - Filters out the extreme outlier file (1064_IEO_DIS_MD.npz with 927 frames)
   - Uses variable-length processing with masking to handle different file lengths

This script orchestrates the full pipeline:
1. Process RAVDESS videos with fixed feature extraction
2. Process CREMA-D videos with fixed feature extraction
3. Train the model with the 6-class architecture
"""

import os
import sys
import subprocess
import logging
import shutil
import argparse
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("run_fixed_pipeline.log"),
        logging.StreamHandler()
    ]
)

def setup_directories():
    """Create the necessary directories for the pipeline."""
    dirs = [
        "processed_ravdess_fixed",
        "processed_crema_d_fixed",
        "models/branched_6class",
        "model_evaluation/branched_6class"
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        logging.info(f"Created directory: {d}")

def process_ravdess(ravdess_dir=None, skip_video=False):
    """Process RAVDESS videos with the fixed preprocessing pipeline."""
    logging.info("Processing RAVDESS dataset with fixed audio feature extraction")
    
    # Use default path if not provided
    if ravdess_dir is None:
        ravdess_dir = "data/RAVDESS"
        logging.info(f"Using default RAVDESS path: {ravdess_dir}")
    
    # Run the fixed preprocessing script
    cmd = [
        "python", "scripts/multimodal_preprocess_fixed.py",
        ravdess_dir,
        "processed_ravdess_fixed"
    ]
    
    # Add skip-video option if requested
    if skip_video:
        cmd.append("--skip-video")
        logging.info("Skipping video feature extraction (using dummy video features)")
    
    try:
        subprocess.run(cmd, check=True)
        logging.info("RAVDESS processing completed successfully")
    except subprocess.CalledProcessError as e:
        logging.error(f"RAVDESS processing failed: {e}")
        return False
    
    return True

def process_crema_d(crema_d_dir=None, skip_video=False):
    """Process CREMA-D videos with the fixed preprocessing pipeline."""
    logging.info("Processing CREMA-D dataset with fixed audio feature extraction")
    
    # Use default path if not provided
    if crema_d_dir is None:
        crema_d_dir = "data/CREMA-D"
        logging.info(f"Using default CREMA-D path: {crema_d_dir}")
    
    # Create a filtered list that excludes the outlier file
    outlier_file = "1064_IEO_DIS_MD.flv"
    logging.info(f"Will exclude outlier file: {outlier_file}")
    
    # Run the fixed preprocessing script
    cmd = [
        "python", "scripts/multimodal_preprocess_fixed.py",
        crema_d_dir, 
        "processed_crema_d_fixed"
    ]
    
    # Add skip-video option if requested
    if skip_video:
        cmd.append("--skip-video")
        logging.info("Skipping video feature extraction (using dummy video features)")
    
    try:
        subprocess.run(cmd, check=True)
        logging.info("CREMA-D processing completed successfully")
    except subprocess.CalledProcessError as e:
        logging.error(f"CREMA-D processing failed: {e}")
        return False
    
    # Extra step: Remove the outlier file if it exists
    outlier_path = os.path.join("processed_crema_d_fixed", outlier_file.replace(".flv", ".npz"))
    if os.path.exists(outlier_path):
        os.remove(outlier_path)
        logging.info(f"Removed outlier file: {outlier_path}")
    
    return True

def train_model():
    """Train the 6-class model with the fixed dataset."""
    logging.info("Training the 6-class model")
    
    cmd = [
        "python", "scripts/train_branched_6class.py",
        "--ravdess-dir", "processed_ravdess_fixed",
        "--cremad-dir", "processed_crema_d_fixed",
        "--model-dir", "models/branched_6class",
        "--eval-dir", "model_evaluation/branched_6class",
        "--epochs", "50",
        "--batch-size", "16",
        "--use-masking"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        logging.info("Model training completed successfully")
    except subprocess.CalledProcessError as e:
        logging.error(f"Model training failed: {e}")
        return False
    
    return True

def copy_sample_data():
    """Copy some sample data from existing processed directories for testing."""
    logging.info("Copying sample data for testing")
    
    # Source directories
    src_ravdess = "processed_ravdess_single_segment"
    src_cremad = "processed_crema_d_single_segment"
    
    # Target directories
    tgt_ravdess = "processed_ravdess_fixed"
    tgt_cremad = "processed_crema_d_fixed"
    
    # Create target directories
    os.makedirs(tgt_ravdess, exist_ok=True)
    os.makedirs(tgt_cremad, exist_ok=True)
    
    # Copy a sample of files (limit to 100 files each)
    ravdess_count = 0
    for file in glob.glob(os.path.join(src_ravdess, "*.npz")):
        if ravdess_count >= 100:
            break
        shutil.copy(file, tgt_ravdess)
        ravdess_count += 1
    
    cremad_count = 0
    for file in glob.glob(os.path.join(src_cremad, "*.npz")):
        # Skip the known outlier file
        if "1064_IEO_DIS_MD" in file:
            continue
        if cremad_count >= 100:
            break
        shutil.copy(file, tgt_cremad)
        cremad_count += 1
    
    logging.info(f"Copied {ravdess_count} RAVDESS sample files")
    logging.info(f"Copied {cremad_count} CREMA-D sample files")
    
    return ravdess_count > 0 and cremad_count > 0

def main():
    parser = argparse.ArgumentParser(description="Run the fixed emotion recognition pipeline")
    parser.add_argument("--skip-ravdess", action="store_true", help="Skip RAVDESS processing")
    parser.add_argument("--skip-cremad", action="store_true", help="Skip CREMA-D processing")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training")
    parser.add_argument("--skip-video", action="store_true", help="Skip video feature extraction (use dummy video features)")
    parser.add_argument("--use-sample-data", action="store_true", help="Use sample data from existing processed directories")
    parser.add_argument("--ravdess-dir", type=str, help="Path to RAVDESS videos directory (default: data/RAVDESS)")
    parser.add_argument("--cremad-dir", type=str, help="Path to CREMA-D videos directory (default: data/CREMA-D)")
    
    args = parser.parse_args()
    
    # Setup directories
    setup_directories()
    
    # Process datasets
    if args.use_sample_data:
        logging.info("Using sample data from existing processed directories")
        if copy_sample_data():
            logging.info("Successfully copied sample data")
        else:
            logging.error("Failed to copy sample data. Make sure processed_ravdess_single_segment and processed_crema_d_single_segment exist.")
            return 1
    else:
        if not args.skip_ravdess:
            if not process_ravdess(args.ravdess_dir, skip_video=args.skip_video):
                logging.error("Failed to process RAVDESS dataset. Exiting.")
                return 1
        else:
            logging.info("Skipping RAVDESS processing")
        
        if not args.skip_cremad:
            if not process_crema_d(args.cremad_dir, skip_video=args.skip_video):
                logging.error("Failed to process CREMA-D dataset. Exiting.")
                return 1
        else:
            logging.info("Skipping CREMA-D processing")
    
    # Train model
    if not args.skip_training:
        if not train_model():
            logging.error("Failed to train model. Exiting.")
            return 1
    else:
        logging.info("Skipping model training")
    
    logging.info("Pipeline completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
