#!/usr/bin/env python
import os
import argparse
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Define the Action Units of interest (intensity only)
# These are common AUs used in facial expression analysis.
# Adjust if your OpenFace version or needs differ.
AU_INTENSITY_COLUMNS = [
    ' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', 
    ' AU07_r', ' AU09_r', ' AU10_r', ' AU12_r', ' AU14_r', 
    ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r', 
    ' AU26_r', ' AU45_r'
] # Note the leading space in column names from OpenFace output

def extract_au_intensities_from_csv(openface_csv_path):
    if not os.path.exists(openface_csv_path):
        logger.warning(f"OpenFace CSV not found: {openface_csv_path}")
        return None
    try:
        df = pd.read_csv(openface_csv_path)
        # Ensure all required AU columns exist
        missing_cols = [col for col in AU_INTENSITY_COLUMNS if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing AU columns in {openface_csv_path}: {missing_cols}. Skipping file.")
            return None
            
        au_data = df[AU_INTENSITY_COLUMNS].values.astype(np.float32)
        if au_data.shape[0] == 0: # No frames detected
             logger.warning(f"No frames with AU data in {openface_csv_path}")
             return None
        return au_data
    except Exception as e:
        logger.error(f"Error processing OpenFace CSV {openface_csv_path}: {e}", exc_info=True)
        return None

def process_manifest_for_video_au_embeddings(
    manifest_df,
    embedding_dir,
    openface_base_dir, # Base directory where OpenFace outputs are stored
    id_col='talk_id',
    video_path_col='raw_video_path', # Used for logging/verification if needed
    overwrite=False
):
    os.makedirs(embedding_dir, exist_ok=True)
    logger.info(f"Saving sequential video AU intensity embeddings to: {embedding_dir}")

    num_expected_aus = len(AU_INTENSITY_COLUMNS)

    for idx, row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="Extracting Sequential Video AU Embeddings"):
        clip_id = str(row[id_col])
        output_filename = f"{clip_id}.npy"
        output_path = os.path.join(embedding_dir, output_filename)

        if os.path.exists(output_path) and not overwrite:
            continue

        # Construct path to the pre-computed OpenFace CSV
        # Assuming a structure like: {openface_base_dir}/{clip_id}/{clip_id}.csv
        # Or simply {openface_base_dir}/{clip_id}.csv if videos are processed directly
        # For UR-FUNNY, talk_id might be like "urfunny_1_1_1", and video file is "1_1_1.mp4"
        # Let's assume the OpenFace output CSV is named after the talk_id directly for simplicity here.
        # This might need adjustment based on actual OpenFace output naming.
        
        # Attempt to derive a base video filename from talk_id if it contains urfunny_ prefix
        base_video_name = clip_id
        if clip_id.startswith("urfunny_"):
            parts = clip_id.split('_')
            if len(parts) > 1: # e.g. urfunny_1_1_1 -> 1_1_1
                base_video_name = "_".join(parts[1:]) 
        
        # Common OpenFace output naming convention is often the video filename without extension
        openface_csv_filename = f"{base_video_name}.csv" 
        # Path might be {openface_base_dir}/{base_video_name}/{base_video_name}.csv or just {openface_base_dir}/{base_video_name}.csv
        # Let's try a common structure: {openface_base_dir}/{video_filename_root}/{video_filename_root}.csv
        # For UR-Funny, the video files are directly in a folder, so the CSVs might be too.
        # Let's assume OpenFace output CSVs are named like {talk_id}.csv or {base_video_name}.csv
        # and are directly in a subfolder named after the video or in a general output folder.
        # We will assume a path structure: {openface_base_dir}/{base_video_name}.csv
        
        # More robust: try to find the CSV based on talk_id or derived base_video_name
        potential_csv_name_1 = f"{clip_id}.csv"
        potential_csv_name_2 = f"{base_video_name}.csv"

        openface_csv_path = os.path.join(openface_base_dir, potential_csv_name_1)
        if not os.path.exists(openface_csv_path):
            openface_csv_path = os.path.join(openface_base_dir, potential_csv_name_2)
            if not os.path.exists(openface_csv_path):
                 # Try one more common pattern: {openface_base_dir}/{base_video_name}/{base_video_name}.csv
                openface_csv_path_alt = os.path.join(openface_base_dir, base_video_name, f"{base_video_name}.csv")
                if os.path.exists(openface_csv_path_alt):
                    openface_csv_path = openface_csv_path_alt
                else:
                    logger.warning(f"OpenFace CSV not found for {clip_id} (tried {potential_csv_name_1}, {potential_csv_name_2}, and nested). Skipping.")
                    np.save(output_path, np.zeros((0, num_expected_aus), dtype=np.float32)) # Save empty if not found
                    continue
        
        au_intensities = extract_au_intensities_from_csv(openface_csv_path)
        
        if au_intensities is None:
            logger.warning(f"Failed to get AU intensities for {clip_id}. Saving empty array.")
            # Save an empty array (0 frames, N AUs) or a specific placeholder
            np.save(output_path, np.zeros((0, num_expected_aus), dtype=np.float32))
        else:
            np.save(output_path, au_intensities)

def main():
    parser = argparse.ArgumentParser(description='Extract sequential OpenFace AU intensity features for UR-FUNNY.')
    parser.add_argument('--manifest_path', type=str, required=True, 
                        help='Path to the input manifest CSV (e.g., datasets/manifests/humor/urfunny_raw_data_complete.csv).')
    parser.add_argument('--embedding_dir', type=str, required=True, 
                        help='Directory to save extracted AU embeddings (e.g., datasets/humor_embeddings_v2/video_sequential_openface_au/).')
    parser.add_argument('--openface_base_dir', type=str, required=True,
                        help='Base directory containing pre-computed OpenFace output CSV files.')
    parser.add_argument('--id_col', type=str, default='talk_id', 
                        help='Column name for unique IDs in the manifest.')
    parser.add_argument('--video_path_col', type=str, default='raw_video_path',
                        help='Column name for video paths (for logging/reference).')
    parser.add_argument('--overwrite', action='store_true', 
                        help='Overwrite existing embeddings.')
    
    args = parser.parse_args()

    try:
        manifest_df = pd.read_csv(args.manifest_path)
        logger.info(f"Loaded manifest: {args.manifest_path} with {len(manifest_df)} entries.")
    except Exception as e:
        logger.error(f"Failed to load manifest file {args.manifest_path}: {e}", exc_info=True)
        sys.exit(1)

    process_manifest_for_video_au_embeddings(
        manifest_df,
        args.embedding_dir,
        args.openface_base_dir,
        id_col=args.id_col,
        video_path_col=args.video_path_col,
        overwrite=args.overwrite
    )
    logger.info("Sequential video AU intensity embedding extraction complete.")

if __name__ == '__main__':
    main()
