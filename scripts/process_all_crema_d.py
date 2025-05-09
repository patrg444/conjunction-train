#!/usr/bin/env python3
"""
Script to process all downsampled CREMA-D files for feature extraction.
This builds on our successful test with a single file and utilizes the multimodal_preprocess_fixed.py module.
"""

import os
import sys
import logging
import time
import argparse
from tqdm import tqdm
import numpy as np
import concurrent.futures
from multimodal_preprocess_fixed import process_video_for_single_segment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("process_all_crema_d.log"),
        logging.StreamHandler()
    ]
)

def process_file(args):
    """Process a single video file. This function is used for multiprocessing."""
    video_path, output_dir, model_name, detector_backend = args
    
    # Check if output file already exists (to resume interrupted processing)
    file_basename = os.path.splitext(os.path.basename(video_path))[0]
    output_file_path = os.path.join(output_dir, f"{file_basename}.npz")
    
    if os.path.exists(output_file_path):
        # File already processed, skip it
        return (True, video_path, f"Already processed: {output_file_path}")
    
    try:
        # Process the video file with additional error handling
        output_file = process_video_for_single_segment(
            video_path=video_path,
            output_dir=output_dir,
            model_name=model_name,
            detector_backend=detector_backend
        )
        
        if output_file:
            return (True, video_path, output_file)
        else:
            return (False, video_path, "No output file generated")
    
    except Exception as e:
        import traceback
        error_msg = f"Error processing {video_path}: {str(e)}\n{traceback.format_exc()}"
        return (False, video_path, error_msg)

def process_all_files(input_dir, output_dir, model_name="VGG-Face", detector_backend="mtcnn", limit=None, num_workers=4, skip_existing=True):
    """Process all video files in a directory."""
    # Ensure input directory exists
    if not os.path.exists(input_dir):
        logging.error(f"Input directory does not exist: {input_dir}")
        print(f"The input directory {input_dir} does not exist. Please check the path.")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Processing files from: {input_dir}")
    logging.info(f"Output directory: {output_dir}")
    
    # Get list of video files with .flv extension
    video_files = []
    for file in os.listdir(input_dir):
        if file.endswith(".flv"):
            video_files.append(os.path.join(input_dir, file))
    
    # Sort files for consistent processing order
    video_files.sort()
    
    # Apply limit if specified
    if limit and limit > 0:
        video_files = video_files[:limit]
        logging.info(f"Processing the first {limit} files")
    
    # Log the number of files to process
    logging.info(f"Found {len(video_files)} video files to process")
    
    # Process files using a process pool for parallelism
    start_time = time.time()
    processed_count = 0
    failed_count = 0
    failed_files = []
    
    # Prepare arguments for each file
    process_args = [(video_file, output_dir, model_name, detector_backend) for video_file in video_files]
    
    # Allow exactly 10 workers as requested by the user
    # This may use more system resources but can speed up processing significantly
    if num_workers > 10:
        logging.warning(f"Reducing worker count from {num_workers} to 10 as requested")
        num_workers = 10
    
    # Use ThreadPoolExecutor for I/O-bound tasks (file operations)
    # This is better than ProcessPoolExecutor for this case as it avoids
    # serialization issues with the large model and data
    logging.info(f"Processing files with {num_workers} workers")
    completed_count = 0
    failed_count = 0
    failed_files = []
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks and get a map of future to filename for tracking
            future_to_file = {executor.submit(process_file, args): args[0] for args in process_args}
            
            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_file), 
                             total=len(video_files), 
                             desc="Processing CREMA-D files"):
                video_file = future_to_file[future]
                try:
                    success, file_path, result = future.result()
                    if success:
                        processed_count += 1
                        logging.info(f"Successfully processed: {file_path}")
                    else:
                        failed_count += 1
                        failed_files.append(file_path)
                        logging.error(f"Failed to process: {file_path}")
                        logging.error(result)
                except Exception as e:
                    failed_count += 1
                    failed_files.append(video_file)
                    logging.error(f"Error in future processing {video_file}: {str(e)}")
                    import traceback
                    logging.error(traceback.format_exc())
    
    except KeyboardInterrupt:
        logging.warning("Process interrupted by user. Will save progress.")
    
    finally:
        # Log processing summary
        elapsed_time = time.time() - start_time
        logging.info(f"Processing status:")
        logging.info(f"Processed {processed_count} files successfully")
        logging.info(f"Failed to process {failed_count} files")
        logging.info(f"Total elapsed time: {elapsed_time:.2f} seconds")
        
        if video_files:
            avg_time = elapsed_time / max(1, processed_count + failed_count)
            logging.info(f"Average time per file: {avg_time:.2f} seconds")
            
            # Estimate remaining time
            remaining_files = len(video_files) - (processed_count + failed_count)
            if remaining_files > 0:
                est_remaining_time = remaining_files * avg_time
                hours = int(est_remaining_time // 3600)
                minutes = int((est_remaining_time % 3600) // 60)
                logging.info(f"Approximately {remaining_files} files remaining")
                logging.info(f"Estimated remaining time: {hours}h {minutes}m")
        
        if failed_count > 0:
            logging.info("Failed files:")
            for failed_file in failed_files:
                logging.info(f"  {failed_file}")
        
        print(f"\nProcessing status:")
        print(f"Processed {processed_count} files successfully")
        print(f"Failed to process {failed_count} files")
        print(f"Total elapsed time: {elapsed_time:.2f} seconds")
        print(f"Results saved to: {output_dir}")
    
    return processed_count, failed_count

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process all downsampled CREMA-D files for feature extraction")
    parser.add_argument("--input-dir", default="downsampled_videos/CREMA-D-audio-complete", 
                      help="Directory containing downsampled CREMA-D videos")
    parser.add_argument("--output-dir", default="crema_d_features", 
                      help="Output directory for processed features")
    parser.add_argument("--model-name", default="VGG-Face", 
                      help="Model name for DeepFace feature extraction")
    parser.add_argument("--detector", default="mtcnn", 
                      help="Face detector backend for DeepFace (e.g., mtcnn, retinaface, opencv)")
    parser.add_argument("--limit", type=int, default=None, 
                      help="Limit processing to the first N files (for testing)")
    parser.add_argument("--workers", type=int, default=4,
                      help="Number of worker processes to use for parallel processing")
    parser.add_argument("--force", action="store_true", 
                      help="Force processing even if output files already exist")
    
    args = parser.parse_args()
    
    # Process all files
    process_all_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        detector_backend=args.detector,
        limit=args.limit,
        num_workers=args.workers,
        skip_existing=not args.force
    )

if __name__ == "__main__":
    main()
