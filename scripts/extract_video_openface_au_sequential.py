#!/usr/bin/env python
import os
import argparse
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
import sys
import subprocess
import tempfile
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Define the Action Units to extract (intensity values)
# These are common AUs, adjust as needed. OpenFace provides AU01_r to AU45_r.
# Example: Basic set often includes AU1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 24, 25, 26
# We will select a subset of these.
SELECTED_AUS_INTENSITY = [
    'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r',
    'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU24_r', 'AU25_r', 'AU26_r'
    # 'AU28_r' # Lip suck often not available or reliable
    # 'AU45_r' # Blink often not available or reliable
]
# Can also add _c for presence if desired, and concatenate

def run_openface(video_path, openface_dir, output_dir_for_openface_csv):
    """
    Runs OpenFace FeatureExtraction on a single video file.
    Saves the output CSV in a specified temporary directory.
    Returns the path to the output CSV file.
    """
    openface_executable = os.path.join(openface_dir, "FeatureExtraction")
    if not os.path.isfile(openface_executable):
        logger.error(f"OpenFace FeatureExtraction not found at {openface_executable}")
        return None

    # Ensure the OpenFace output directory for this specific video exists
    # The CSV will be named after the video file inside this directory
    os.makedirs(output_dir_for_openface_csv, exist_ok=True)

    cmd = [
        openface_executable,
        "-f", video_path,
        "-out_dir", output_dir_for_openface_csv, # OpenFace will create a subdir here if it doesn't exist
        "-aus",      # Output Action Units
        "-quiet"     # Suppress OpenFace GUI and verbose console output
    ]
    logger.info(f"Running OpenFace: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, timeout=300) # 5 min timeout
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        # OpenFace creates a CSV in a subdirectory named after the video file, without extension
        # However, with -out_dir, it places the CSV directly in that dir.
        # Let's assume it's directly in output_dir_for_openface_csv
        output_csv_path = os.path.join(output_dir_for_openface_csv, f"{video_basename}.csv")

        if os.path.exists(output_csv_path):
            return output_csv_path
        else:
            # Fallback: check if OpenFace created a subdirectory (older behavior for some versions)
            older_output_csv_path = os.path.join(output_dir_for_openface_csv, video_basename, f"{video_basename}.csv")
            if os.path.exists(older_output_csv_path):
                return older_output_csv_path
            logger.error(f"OpenFace output CSV not found at {output_csv_path} or {older_output_csv_path} for video {video_path}")
            return None
    except subprocess.CalledProcessError as e:
        logger.error(f"OpenFace failed for video {video_path}: {e}")
        return None
    except subprocess.TimeoutExpired:
        logger.error(f"OpenFace timed out for video {video_path}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while running OpenFace for {video_path}: {e}", exc_info=True)
        return None

def extract_au_features_from_csv(csv_path, selected_aus):
    try:
        df = pd.read_csv(csv_path)
        # OpenFace CSVs have leading/trailing spaces in column names
        df.columns = df.columns.str.strip()
        
        # Check if all selected AUs are present
        missing_aus = [au for au in selected_aus if au not in df.columns]
        if missing_aus:
            logger.warning(f"Missing AUs in {csv_path}: {missing_aus}. Available: {list(df.columns)}")
            # Proceed with available AUs, or return None/zeros
            # For simplicity, let's filter to only available AUs from our selection
            selected_aus = [au for au in selected_aus if au in df.columns]
            if not selected_aus:
                logger.error(f"No selected AUs found in {csv_path}. Cannot extract features.")
                return None
        
        au_features = df[selected_aus].values.astype(np.float32)
        return au_features # Shape: (num_frames, num_selected_AUs)
    except Exception as e:
        logger.error(f"Error processing OpenFace CSV {csv_path}: {e}", exc_info=True)
        return None

def process_manifest_videos(
    manifest_df,
    openface_dir, # Path to OpenFace build/bin directory
    embedding_dir, # Where to save final .npy AU sequences
    video_path_col='video_path', # Column in manifest with video file paths
    id_col='talk_id',
    overwrite=False
):
    os.makedirs(embedding_dir, exist_ok=True)
    logger.info(f"Saving sequential OpenFace AU embeddings to: {embedding_dir}")

    # Create a single temporary directory for all OpenFace CSV outputs
    # This avoids creating too many subdirectories if OpenFace is run many times.
    # OpenFace will create subdirectories within this if needed.
    temp_openface_output_parent_dir = tempfile.mkdtemp(prefix="openface_outputs_")
    logger.info(f"Temporary OpenFace CSVs will be stored under: {temp_openface_output_parent_dir}")

    try:
        for idx, row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="Extracting OpenFace AU Features"):
            clip_id = str(row[id_col])
            output_filename = f"{clip_id}.npy"
            output_path = os.path.join(embedding_dir, output_filename)

            if os.path.exists(output_path) and not overwrite:
                continue

            if video_path_col not in row or not isinstance(row[video_path_col], str) or not row[video_path_col].strip():
                logger.warning(f"Missing or invalid video path for {clip_id} in column '{video_path_col}'. Skipping.")
                np.save(output_path, np.zeros((1, len(SELECTED_AUS_INTENSITY)), dtype=np.float32)) # Save dummy
                continue
            
            video_file_path = row[video_path_col].strip()
            if not os.path.exists(video_file_path):
                logger.warning(f"Video file not found for {clip_id} at {video_file_path}. Skipping.")
                np.save(output_path, np.zeros((1, len(SELECTED_AUS_INTENSITY)), dtype=np.float32)) # Save dummy
                continue

            # Define a specific output directory for this video's OpenFace CSV
            # This helps keep OpenFace outputs organized if it creates subdirs by video name
            video_basename_for_dir = os.path.splitext(os.path.basename(video_file_path))[0]
            specific_openface_csv_out_dir = os.path.join(temp_openface_output_parent_dir, video_basename_for_dir)
            os.makedirs(specific_openface_csv_out_dir, exist_ok=True)


            openface_csv_path = run_openface(video_file_path, openface_dir, specific_openface_csv_out_dir)

            if openface_csv_path and os.path.exists(openface_csv_path):
                au_sequence = extract_au_features_from_csv(openface_csv_path, SELECTED_AUS_INTENSITY)
                if au_sequence is not None and au_sequence.ndim == 2 and au_sequence.shape[0] > 0:
                    np.save(output_path, au_sequence)
                else:
                    logger.warning(f"Failed to extract valid AU sequence for {clip_id}. Saving zeros.")
                    np.save(output_path, np.zeros((1, len(SELECTED_AUS_INTENSITY)), dtype=np.float32))
                # Clean up the specific OpenFace CSV output for this video
                # shutil.rmtree(specific_openface_csv_out_dir, ignore_errors=True) # If OpenFace creates subdirs
                if os.path.exists(openface_csv_path): # If CSV is directly in specific_openface_csv_out_dir
                    try: os.remove(openface_csv_path)
                    except OSError: pass

            else:
                logger.warning(f"OpenFace CSV not generated for {clip_id}. Saving zeros.")
                np.save(output_path, np.zeros((1, len(SELECTED_AUS_INTENSITY)), dtype=np.float32))
    finally:
        # Clean up the parent temporary directory for OpenFace outputs
        shutil.rmtree(temp_openface_output_parent_dir, ignore_errors=True)
        logger.info(f"Cleaned up temporary OpenFace output directory: {temp_openface_output_parent_dir}")


def main():
    parser = argparse.ArgumentParser(description='Extract sequential Action Unit features using OpenFace.')
    parser.add_argument('--manifest_path', type=str, required=True, help='Path to the input manifest CSV (must contain video paths and IDs).')
    parser.add_argument('--openface_dir', type=str, required=True, help='Path to the OpenFace build/bin directory (containing FeatureExtraction).')
    parser.add_argument('--embedding_dir', type=str, required=True, help='Directory to save extracted sequential AU embeddings (.npy files).')
    parser.add_argument('--video_path_col', type=str, default='video_path', help='Column name in manifest for video file paths.')
    parser.add_argument('--id_col', type=str, default='talk_id', help='Column name for unique IDs.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing embeddings.')
    
    args = parser.parse_args()

    try:
        manifest_df = pd.read_csv(args.manifest_path)
        logger.info(f"Loaded manifest: {args.manifest_path} with {len(manifest_df)} entries.")
    except Exception as e:
        logger.error(f"Failed to load manifest file {args.manifest_path}: {e}", exc_info=True)
        sys.exit(1)

    process_manifest_videos(
        manifest_df,
        args.openface_dir,
        args.embedding_dir,
        video_path_col=args.video_path_col,
        id_col=args.id_col,
        overwrite=args.overwrite
    )
    logger.info("OpenFace AU feature extraction complete.")

if __name__ == '__main__':
    main()
