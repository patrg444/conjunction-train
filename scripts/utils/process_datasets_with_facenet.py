#!/usr/bin/env python3
"""
Script to process RAVDESS and CREMA-D datasets using the FaceNet extractor.
This script leverages the multimodal_preprocess_fixed.py module with FaceNet instead of DeepFace.
"""

import os
import sys
import logging
import time
import argparse
from tqdm import tqdm
import numpy as np
import concurrent.futures
from multimodal_preprocess_fixed import process_video_for_single_segment, process_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("process_datasets_with_facenet.log"),
        logging.StreamHandler()
    ]
)

def process_ravdess_dataset(input_dir="downsampled_videos/RAVDESS", 
                            output_dir="ravdess_features_facenet", 
                            num_workers=4, 
                            limit=None):
    """Process the RAVDESS dataset using FaceNet extractor."""
    logging.info(f"Processing RAVDESS dataset from {input_dir}")
    
    if not os.path.exists(input_dir):
        logging.error(f"RAVDESS directory not found: {input_dir}")
        return 0, 0
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get video paths
    video_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".mp4"):
                video_paths.append(os.path.join(root, file))
    
    # Sort paths for consistent processing
    video_paths.sort()
    
    # Apply limit if specified
    if limit and limit > 0:
        video_paths = video_paths[:limit]
        logging.info(f"Processing only the first {limit} RAVDESS videos")
    
    logging.info(f"Found {len(video_paths)} RAVDESS videos to process")
    
    # Process files using a thread pool
    start_time = time.time()
    processed_count = 0
    failed_count = 0
    failed_files = []
    
    # Prepare arguments for each file
    process_args = [(video_path, output_dir, None, "mtcnn", "facenet", False, None, input_dir) 
                    for video_path in video_paths]
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Map function to all files
            futures = [executor.submit(process_video_for_single_segment, *args) for args in process_args]
            
            # Process results as they complete
            for i, future in enumerate(tqdm(concurrent.futures.as_completed(futures), 
                                         total=len(futures), 
                                         desc="Processing RAVDESS videos with FaceNet")):
                video_path = video_paths[i]
                try:
                    result = future.result()
                    if result:
                        processed_count += 1
                        logging.info(f"Successfully processed: {video_path}")
                    else:
                        failed_count += 1
                        failed_files.append(video_path)
                        logging.error(f"Failed to process: {video_path}")
                except Exception as e:
                    failed_count += 1
                    failed_files.append(video_path)
                    logging.error(f"Error processing {video_path}: {str(e)}")
    
    except KeyboardInterrupt:
        logging.warning("Process interrupted by user")
    
    # Log processing summary
    elapsed_time = time.time() - start_time
    logging.info(f"RAVDESS Processing Summary:")
    logging.info(f"Processed {processed_count} videos successfully")
    logging.info(f"Failed to process {failed_count} videos")
    logging.info(f"Total elapsed time: {elapsed_time:.2f} seconds")
    
    return processed_count, failed_count

def process_crema_d_dataset(input_dir="downsampled_videos/CREMA-D-audio-complete", 
                           output_dir="crema_d_features_facenet", 
                           num_workers=4, 
                           limit=None):
    """Process the CREMA-D dataset using FaceNet extractor."""
    logging.info(f"Processing CREMA-D dataset from {input_dir}")
    
    if not os.path.exists(input_dir):
        logging.error(f"CREMA-D directory not found: {input_dir}")
        return 0, 0
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get video paths (CREMA-D uses .flv files)
    video_paths = []
    for file in os.listdir(input_dir):
        if file.endswith(".flv"):
            video_paths.append(os.path.join(input_dir, file))
    
    # Sort paths for consistent processing
    video_paths.sort()
    
    # Apply limit if specified
    if limit and limit > 0:
        video_paths = video_paths[:limit]
        logging.info(f"Processing only the first {limit} CREMA-D videos")
    
    logging.info(f"Found {len(video_paths)} CREMA-D videos to process")
    
    # Process files using a thread pool
    start_time = time.time()
    processed_count = 0
    failed_count = 0
    failed_files = []
    
    # Prepare arguments for each file
    process_args = [(video_path, output_dir, None, "mtcnn", "facenet", False, None, input_dir) 
                    for video_path in video_paths]
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Map function to all files
            futures = [executor.submit(process_video_for_single_segment, *args) for args in process_args]
            
            # Process results as they complete
            for i, future in enumerate(tqdm(concurrent.futures.as_completed(futures), 
                                         total=len(futures), 
                                         desc="Processing CREMA-D videos with FaceNet")):
                video_path = video_paths[i]
                try:
                    result = future.result()
                    if result:
                        processed_count += 1
                        logging.info(f"Successfully processed: {video_path}")
                    else:
                        failed_count += 1
                        failed_files.append(video_path)
                        logging.error(f"Failed to process: {video_path}")
                except Exception as e:
                    failed_count += 1
                    failed_files.append(video_path)
                    logging.error(f"Error processing {video_path}: {str(e)}")
    
    except KeyboardInterrupt:
        logging.warning("Process interrupted by user")
    
    # Log processing summary
    elapsed_time = time.time() - start_time
    logging.info(f"CREMA-D Processing Summary:")
    logging.info(f"Processed {processed_count} videos successfully")
    logging.info(f"Failed to process {failed_count} videos")
    logging.info(f"Total elapsed time: {elapsed_time:.2f} seconds")
    
    return processed_count, failed_count

def process_both_datasets(ravdess_input="downsampled_videos/RAVDESS",
                         ravdess_output="ravdess_features_facenet",
                         crema_d_input="downsampled_videos/CREMA-D-audio-complete",
                         crema_d_output="crema_d_features_facenet",
                         num_workers=4,
                         limit=None):
    """Process both RAVDESS and CREMA-D datasets sequentially."""
    # Process RAVDESS dataset
    logging.info("Starting RAVDESS dataset processing...")
    ravdess_success, ravdess_failed = process_ravdess_dataset(
        input_dir=ravdess_input,
        output_dir=ravdess_output,
        num_workers=num_workers,
        limit=limit
    )
    
    # Process CREMA-D dataset
    logging.info("Starting CREMA-D dataset processing...")
    crema_d_success, crema_d_failed = process_crema_d_dataset(
        input_dir=crema_d_input,
        output_dir=crema_d_output,
        num_workers=num_workers,
        limit=limit
    )
    
    # Log combined results
    logging.info("Combined Processing Summary:")
    logging.info(f"RAVDESS: {ravdess_success} successes, {ravdess_failed} failures")
    logging.info(f"CREMA-D: {crema_d_success} successes, {crema_d_failed} failures")
    logging.info(f"Total: {ravdess_success + crema_d_success} successes, {ravdess_failed + crema_d_failed} failures")
    
    return (ravdess_success, ravdess_failed, crema_d_success, crema_d_failed)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process RAVDESS and CREMA-D datasets with FaceNet")
    parser.add_argument("--ravdess-only", action="store_true", help="Process only RAVDESS dataset")
    parser.add_argument("--crema-d-only", action="store_true", help="Process only CREMA-D dataset")
    parser.add_argument("--ravdess-input", default="downsampled_videos/RAVDESS", 
                      help="Directory containing RAVDESS videos")
    parser.add_argument("--ravdess-output", default="ravdess_features_facenet", 
                      help="Output directory for RAVDESS features")
    parser.add_argument("--crema-d-input", default="downsampled_videos/CREMA-D-audio-complete", 
                      help="Directory containing CREMA-D videos")
    parser.add_argument("--crema-d-output", default="crema_d_features_facenet", 
                      help="Output directory for CREMA-D features")
    parser.add_argument("--workers", type=int, default=4,
                      help="Number of worker threads (default: 4)")
    parser.add_argument("--limit", type=int, default=None,
                      help="Limit processing to first N videos per dataset (for testing)")
    
    args = parser.parse_args()
    
    # Process based on arguments
    if args.ravdess_only:
        process_ravdess_dataset(
            input_dir=args.ravdess_input,
            output_dir=args.ravdess_output,
            num_workers=args.workers,
            limit=args.limit
        )
    elif args.crema_d_only:
        process_crema_d_dataset(
            input_dir=args.crema_d_input,
            output_dir=args.crema_d_output,
            num_workers=args.workers,
            limit=args.limit
        )
    else:
        # Process both datasets by default
        process_both_datasets(
            ravdess_input=args.ravdess_input,
            ravdess_output=args.ravdess_output,
            crema_d_input=args.crema_d_input,
            crema_d_output=args.crema_d_output,
            num_workers=args.workers,
            limit=args.limit
        )

if __name__ == "__main__":
    main()
