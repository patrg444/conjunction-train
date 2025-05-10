#!/usr/bin/env python3
"""
Batch downsampling script that preprocesses all videos to 15 fps.
This creates optimized videos for subsequent feature extraction.
"""

import os
import sys
import glob
import argparse
import logging
import time
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

# Import our existing resample_video function
from multimodal_preprocess_fixed import resample_video

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("batch_downsample.log"),
        logging.StreamHandler()
    ]
)

def setup_directories(input_dir, output_dir):
    """Setup and validate directories."""
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")
    
    return input_dir, output_dir

def find_videos(input_dir, patterns=["*.mp4", "*.flv"]):
    """Find all video files matching the given patterns."""
    all_videos = []
    all_found_videos = []  # Track all videos found before filtering
    filtered_videos = []   # Track filtered videos for logging
    
    # Check if input_dir is itself a video file
    if any(input_dir.endswith(ext) for ext in ['.mp4', '.flv']):
        if os.path.exists(input_dir):
            return [input_dir], [], 0
        else:
            logging.error(f"Input video file does not exist: {input_dir}")
            return [], [], 0
    
    # Use just a single method for finding videos to avoid duplicates
    for pattern in patterns:
        # Check if we should use recursive search
        if "RAVDESS" in input_dir or os.path.isdir(os.path.join(input_dir, "Actor_01")):
            # For nested directory structure (e.g., RAVDESS)
            for root, _, _ in os.walk(input_dir):
                nested_matches = glob.glob(os.path.join(root, pattern))
                all_found_videos.extend(nested_matches)
                
                # Filter out any files that are in a temp directory or duplicates
                new_matches = []
                for m in nested_matches:
                    if "temp_resampled_videos" in m:
                        filtered_videos.append((m, "temp_directory"))
                    elif m in all_videos:
                        filtered_videos.append((m, "duplicate"))
                    else:
                        new_matches.append(m)
                
                all_videos.extend(new_matches)
        else:
            # For flat directory structure like CREMA-D
            direct_matches = glob.glob(os.path.join(input_dir, pattern))
            all_found_videos.extend(direct_matches)
            
            # Filter out any files that are in a temp directory or duplicates
            new_matches = []
            for m in direct_matches:
                if "temp_resampled_videos" in m:
                    filtered_videos.append((m, "temp_directory"))
                elif m in all_videos:
                    filtered_videos.append((m, "duplicate"))
                else:
                    new_matches.append(m)
            
            all_videos.extend(new_matches)
    
    # Log filtering statistics
    total_found = len(all_found_videos)
    duplicates = sum(1 for _, reason in filtered_videos if reason == "duplicate")
    temp_files = sum(1 for _, reason in filtered_videos if reason == "temp_directory")
    
    logging.info(f"Found {total_found} total videos matching patterns {patterns}")
    logging.info(f"Filtered out {duplicates} duplicate videos")
    logging.info(f"Filtered out {temp_files} videos in temporary directories")
    logging.info(f"Final unique video count: {len(all_videos)}")
    
    return all_videos, filtered_videos, len(all_found_videos)

def create_output_path(video_path, input_dir, output_dir):
    """Create the output path maintaining the original directory structure."""
    # Check if input_dir is a file
    if os.path.isfile(input_dir):
        # For single file input, just use the filename in the output directory
        return os.path.join(output_dir, os.path.basename(video_path))
    
    # For directory input, safely preserve structure without .. components
    try:
        rel_path = os.path.relpath(video_path, input_dir)
        # Catch paths that try to go up directories with '..'
        if '..' in rel_path:
            # Just use the basename as fallback
            rel_path = os.path.basename(video_path)
    except ValueError:
        # If paths are on different drives or otherwise incompatible
        rel_path = os.path.basename(video_path)
    
    output_path = os.path.join(output_dir, rel_path)
    
    # Ensure the directory exists
    output_dir_path = os.path.dirname(output_path)
    os.makedirs(output_dir_path, exist_ok=True)
    
    return output_path

def downsample_video(args):
    """Downsample a single video to the target FPS."""
    video_path, input_dir, output_dir, fps = args
    
    # Create output path with original directory structure
    output_path = create_output_path(video_path, input_dir, output_dir)
    
    # Skip if the output file already exists
    if os.path.exists(output_path):
        logging.info(f"Output video already exists, skipping: {output_path}")
        return output_path, 0, True  # Path, processing time, skipped flag
    
    try:
        # Use the resample_video function but specify our own output path
        start_time = time.time()
        
        # Get the directory where the video is to be saved
        output_dir_path = os.path.dirname(output_path)
        
        # Extract just the filename for the resampled video
        output_filename = os.path.basename(output_path)
        
        # Ensure directories exist without changing working directory
        os.makedirs(output_dir_path, exist_ok=True)
        
        # Use absolute paths to avoid working directory issues
        video_path_abs = os.path.abspath(video_path)
        
        # Create a temporary directory for the resampled video
        temp_output = resample_video(video_path_abs, fps=fps)
        
        # If temp_output is not in the expected location, copy it
        if temp_output != output_path:
            # Use os.rename instead of shutil.copy to avoid copying the same file
            try:
                # Check if temporary file exists (use the full path)
                if os.path.exists(temp_output):
                    # Ensure the output directory exists
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    # Copy the file to the desired output location
                    import shutil
                    shutil.copy2(temp_output, output_path)
                    logging.info(f"Copied {temp_output} to {output_path}")
                else:
                    logging.warning(f"Could not find temporary file: {temp_output}")
            except Exception as e:
                logging.error(f"Error copying file: {str(e)}")
                import traceback
                logging.error(traceback.format_exc())
        
        processing_time = time.time() - start_time
        
        # Only report success if the file was actually created
        if os.path.exists(output_path) or os.path.exists(temp_output):
            logging.info(f"Successfully downsampled {video_path} in {processing_time:.2f} seconds")
            return output_path, processing_time, False  # Path, processing time, skipped flag
        else:
            logging.error(f"Failed to create output file for {video_path}")
            return None, 0, False  # Path, processing time, skipped flag
    
    except Exception as e:
        logging.error(f"Error downsampling {video_path}: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None, 0, False  # Path, processing time, skipped flag

def batch_downsample(input_dir, output_dir, fps=15, max_workers=4, limit=None):
    """Batch downsample all videos in input_dir to the specified fps."""
    # Setup and validate directories
    input_dir, output_dir = setup_directories(input_dir, output_dir)
    
    # Find all videos
    all_videos, filtered_videos, total_found = find_videos(input_dir)
    if not all_videos:
        logging.warning(f"No videos found in {input_dir}")
        return []
    
    # Apply limit if specified
    if limit and limit > 0 and limit < len(all_videos):
        logging.info(f"Limiting processing to first {limit} videos (out of {len(all_videos)} total)")
        limited_videos = all_videos[:limit]
    else:
        limited_videos = all_videos
        if limit:
            logging.info(f"Requested limit of {limit} is greater than or equal to total videos ({len(all_videos)}). Processing all videos.")
    
    # Prepare arguments for parallel processing
    process_args = [(video, input_dir, output_dir, fps) for video in limited_videos]
    
    # Process videos in parallel or sequentially
    successful = 0
    skipped = 0
    failed = 0
    output_paths = []
    total_time = 0
    
    if max_workers > 1:
        # Parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Process videos with progress bar
            with tqdm(total=len(limited_videos), desc=f"Downsampling videos to {fps} fps") as progress:
                for output_path, processing_time, was_skipped in executor.map(downsample_video, process_args):
                    if was_skipped:
                        skipped += 1
                    elif output_path:
                        output_paths.append(output_path)
                        successful += 1
                        total_time += processing_time
                    else:
                        failed += 1
                    progress.update(1)
    else:
        # Sequential processing
        for args in tqdm(process_args, desc=f"Downsampling videos to {fps} fps"):
            output_path, processing_time, was_skipped = downsample_video(args)
            if was_skipped:
                skipped += 1
            elif output_path:
                output_paths.append(output_path)
                successful += 1
                total_time += processing_time
            else:
                failed += 1
    
    # Report results
    logging.info(f"Batch downsampling complete:")
    logging.info(f"  Successfully downsampled: {successful}")
    logging.info(f"  Skipped (already existed): {skipped}")
    logging.info(f"  Failed: {failed}")
    
    if successful > 0:
        avg_time = total_time / successful
        logging.info(f"  Average processing time: {avg_time:.2f} seconds per video")
    
    return output_paths

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Batch downsample videos to a target FPS")
    parser.add_argument("input_dir", help="Input directory containing videos or path to a single video")
    parser.add_argument("output_dir", help="Output directory for downsampled videos")
    parser.add_argument("--fps", type=float, default=15, help="Target FPS (default: 15)")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum worker threads (default: 4)")
    parser.add_argument("--limit", type=int, help="Limit processing to the first N videos")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Run batch downsampling
    output_paths = batch_downsample(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        fps=args.fps,
        max_workers=args.max_workers,
        limit=args.limit
    )
    
    print(f"\nSuccessfully downsampled {len(output_paths)} videos to {args.output_dir}")
