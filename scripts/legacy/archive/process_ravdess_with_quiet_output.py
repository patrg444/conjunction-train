#!/usr/bin/env python3
"""
Process RAVDESS dataset files with reduced terminal output.
This script wraps around multimodal_preprocess.py functionality but provides
cleaner progress reporting.
"""

import os
import sys
import glob
import logging
from tqdm import tqdm
import time

# Import functionality from the main processing script
from multimodal_preprocess import (
    process_video_for_multimodal_lstm,
    configure_logging
)

def print_progress(current, total, success_count, last_file=None, error_files=None):
    """Print a concise progress report to the console."""
    success_rate = (success_count / current * 100) if current > 0 else 0
    status_msg = f"Processed: {current}/{total} | Success: {success_count} ({success_rate:.1f}%)"
    
    if last_file:
        status_msg += f" | Last: {os.path.basename(last_file)}"
    
    if error_files and len(error_files) > 0:
        status_msg += f" | Errors: {len(error_files)}"
    
    # Clear line and print new status
    print(f"\r{status_msg}", end="")
    sys.stdout.flush()

def process_ravdess_dataset(input_dir, output_dir="processed_all_ravdess", model_name="VGG-Face"):
    """Process all RAVDESS video files with quiet console output.
    
    Args:
        input_dir: Directory containing RAVDESS video files
        output_dir: Directory to save processed features
        model_name: DeepFace model to use
        
    Returns:
        Tuple of (success_count, total_count, error_files)
    """
    # Set up logging - all INFO to file, only errors to console
    logger = configure_logging(verbose=False)
    
    # Find all mp4 files
    video_paths = glob.glob(os.path.join(input_dir, "*.mp4"))
    
    if not video_paths:
        print(f"No video files found in {input_dir}")
        return 0, 0, []
    
    total_count = len(video_paths)
    print(f"Found {total_count} RAVDESS video files to process")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Track progress
    success_count = 0
    error_files = []
    
    # Process each video
    for i, video_path in enumerate(video_paths):
        try:
            # Process the video
            output_file = process_video_for_multimodal_lstm(
                video_path=video_path,
                output_dir=output_dir,
                model_name=model_name,
                window_size=1.0,    # 1-second segments for RAVDESS
                hop_size=0.5,       # 0.5-second hop size
                sub_window_size=0.2,
                sub_window_hop=0.1
            )
            
            if output_file:
                success_count += 1
                logging.info(f"Successfully processed {video_path} -> {output_file}")
            else:
                error_files.append(video_path)
                logging.error(f"Failed to process {video_path}")
            
        except Exception as e:
            error_files.append(video_path)
            logging.error(f"Error processing {video_path}: {str(e)}")
        
        # Update progress
        print_progress(i+1, total_count, success_count, video_path, error_files)
        
        # Small delay to avoid console flicker
        time.sleep(0.01)
    
    print("\n")  # Add a newline after the progress bar
    
    # Print summary
    print(f"\nProcessing complete!")
    print(f"Successfully processed {success_count} out of {total_count} videos ({success_count/total_count*100:.1f}%)")
    
    if error_files:
        print(f"Encountered errors with {len(error_files)} files. See multimodal_preprocess.log for details.")
    
    return success_count, total_count, error_files

if __name__ == "__main__":
    # Get input directory from command line or use default
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    else:
        print("Usage: python process_ravdess_with_quiet_output.py <path_to_ravdess_videos>")
        sys.exit(1)
    
    # Set output directory
    output_dir = "processed_all_ravdess"
    
    # Process dataset
    start_time = time.time()
    success_count, total_count, error_files = process_ravdess_dataset(input_dir, output_dir)
    elapsed_time = time.time() - start_time
    
    # Print timing information
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total processing time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
    
    # Write error list to file if any errors occurred
    if error_files:
        with open("ravdess_processing_errors.txt", "w") as f:
            for error_file in error_files:
                f.write(f"{error_file}\n")
        print(f"List of error files saved to ravdess_processing_errors.txt")
