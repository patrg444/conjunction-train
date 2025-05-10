import cv2
import glob
import numpy as np
import os
from tqdm import tqdm

# Define the target sampling rate
TARGET_FPS = 15.0

# Define paths to the video directories
# Adjust patterns if needed based on actual file locations/extensions
ravdess_pattern = "downsampled_videos/RAVDESS/Actor_*/*.mp4"
cremad_pattern = "downsampled_videos/CREMA-D-audio-complete/*.mp4" # Assuming mp4, adjust if flv etc.

video_files = glob.glob(ravdess_pattern, recursive=True) + glob.glob(cremad_pattern, recursive=True)

if not video_files:
    print("Error: No video files found in specified directories.")
    print(f"Searched RAVDESS: {ravdess_pattern}")
    print(f"Searched CREMA-D: {cremad_pattern}")
    exit()

print(f"Found {len(video_files)} video files to analyze.")

sampled_frame_counts = []
failed_files = []

for video_path in tqdm(video_files, desc="Analyzing videos"):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video file: {video_path}")
            failed_files.append(video_path)
            continue

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if original_fps > 0 and total_frames > 0:
            # Calculate approximate number of frames after sampling at TARGET_FPS
            # duration = total_frames / original_fps
            # num_sampled = int(round(duration * TARGET_FPS))
            # More direct calculation: keep every Nth frame where N = original_fps / TARGET_FPS
            sampling_ratio = original_fps / TARGET_FPS
            if sampling_ratio <= 0: # Avoid division by zero or weird ratios
                 num_sampled = total_frames # Keep all if original fps is low or zero
            else:
                 num_sampled = int(np.ceil(total_frames / sampling_ratio))

            sampled_frame_counts.append(num_sampled)
        elif total_frames > 0:
             print(f"Warning: Could not get FPS for {video_path}. Using total frames: {total_frames}")
             sampled_frame_counts.append(total_frames) # Fallback if FPS is unavailable
        else:
            print(f"Warning: Could not get frame count or FPS for {video_path}")
            failed_files.append(video_path)

    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        failed_files.append(video_path)
        if 'cap' in locals() and cap.isOpened():
            cap.release()

if not sampled_frame_counts:
    print("Error: No valid frame counts could be extracted.")
    exit()

# Calculate statistics
counts_array = np.array(sampled_frame_counts)
print("\n--- Sampled Frame Count Statistics (at approx 15 fps) ---")
print(f"Min:    {np.min(counts_array)}")
print(f"Max:    {np.max(counts_array)}")
print(f"Mean:   {np.mean(counts_array):.2f}")
print(f"Median: {np.median(counts_array)}")
print(f"50th Percentile (Median): {np.percentile(counts_array, 50)}")
print(f"75th Percentile:          {np.percentile(counts_array, 75)}")
print(f"90th Percentile:          {np.percentile(counts_array, 90)}")
print(f"95th Percentile:          {np.percentile(counts_array, 95)}")
print(f"99th Percentile:          {np.percentile(counts_array, 99)}")

if failed_files:
    print(f"\nWarning: Failed to process {len(failed_files)} files:")
    for f in failed_files[:10]: # Print first 10 failed files
        print(f"- {f}")
    if len(failed_files) > 10:
        print("  ...")
