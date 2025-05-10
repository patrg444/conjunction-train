#!/usr/bin/env python3
"""
Process a small sample of videos from RAVDESS and CREMA-D datasets using the fixed pipeline
to verify that both video and audio features are being extracted correctly.
"""

import os
import glob
import logging
import sys
import random
from multimodal_preprocess_fixed import process_video_for_single_segment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def process_samples(dataset_dir, output_dir, pattern="*.mp4", sample_size=3):
    """Process a sample of videos from the dataset."""
    if not os.path.exists(dataset_dir):
        logging.error(f"Dataset directory not found: {dataset_dir}")
        return False

    # Find all video files
    if "RAVDESS" in dataset_dir:
        # RAVDESS has a different structure - need to look in subdirectories
        video_paths = []
        for actor_dir in glob.glob(os.path.join(dataset_dir, "Actor_*")):
            video_paths.extend(glob.glob(os.path.join(actor_dir, pattern)))
    else:
        # For CREMA-D, videos are in VideoFlash directory
        if "CREMA-D" in dataset_dir and not "VideoFlash" in dataset_dir:
            video_dir = os.path.join(dataset_dir, "VideoFlash")
            if os.path.exists(video_dir):
                video_paths = glob.glob(os.path.join(video_dir, pattern))
            else:
                video_paths = glob.glob(os.path.join(dataset_dir, pattern))
        else:
            video_paths = glob.glob(os.path.join(dataset_dir, pattern))

    if not video_paths:
        logging.error(f"No videos found in {dataset_dir} with pattern {pattern}")
        return False

    # Sample videos
    if len(video_paths) > sample_size:
        sample_videos = random.sample(video_paths, sample_size)
    else:
        sample_videos = video_paths

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each video
    successful = 0
    for video_path in sample_videos:
        logging.info(f"Processing video: {video_path}")
        try:
            output_file = process_video_for_single_segment(
                video_path=video_path,
                output_dir=output_dir
            )
            if output_file:
                successful += 1
                logging.info(f"Successfully processed {os.path.basename(video_path)}")
            else:
                logging.error(f"Failed to process {os.path.basename(video_path)}")
        except Exception as e:
            logging.error(f"Error processing {os.path.basename(video_path)}: {str(e)}")

    # Report results
    logging.info(f"Successfully processed {successful} out of {len(sample_videos)} videos")
    return successful > 0

def main():
    # Process samples from RAVDESS
    ravdess_dir = "data/RAVDESS"
    ravdess_output = "processed_ravdess_fixed"
    ravdess_success = process_samples(ravdess_dir, ravdess_output, pattern="*.mp4", sample_size=3)

    # Process samples from CREMA-D
    crema_d_dir = "data/CREMA-D"
    crema_d_output = "processed_crema_d_fixed"
    crema_d_success = process_samples(crema_d_dir, crema_d_output, pattern="*.flv", sample_size=3)

    # Validate the processed files
    if ravdess_success or crema_d_success:
        logging.info("Processing samples completed successfully")
        print("\nValidating processed files...")
        # Use our validation script to check the processed files
        validation_script = "scripts/validate_feature_extraction.py"
        if os.path.exists(validation_script):
            os.system(f"python {validation_script}")
        else:
            logging.error(f"Validation script not found: {validation_script}")
    else:
        logging.error("Failed to process any samples")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
