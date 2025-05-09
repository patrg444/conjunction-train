import argparse
import csv
import os
import random
import pandas as pd
from pathlib import Path
import logging
import json # Added for SMILE JSON parsing
import glob # Added for listing SMILE segments
from tqdm import tqdm # Added for progress bar
# import subprocess # Potentially needed for ffprobe if calculating duration here

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
TARGET_DURATION_S = 1.0
SAMPLE_RATE = 16000 # Assuming 16kHz for audio processing consistency

def parse_ava(ava_dir, dataset_root):
    """Parses AVA-Laughter annotations and clip structure."""
    logging.info(f"Parsing AVA dataset from: {ava_dir}")
    ava_root = Path(ava_dir)
    dataset_root = Path(dataset_root)
    anno_file = ava_root / "annos" / "laughter_v1.csv"
    clip_dir = Path(ava_dir) / "clips"
    manifest_entries = []

    if not anno_file.exists():
        logging.warning(f"AVA annotation file not found: {anno_file}. Skipping AVA.")
        return []
    if not clip_dir.is_dir():
         logging.warning(f"AVA clip directory not found: {clip_dir}. Skipping AVA.")
         return []

    try:
        with open(anno_file, 'r') as f:
            reader = csv.reader(f)
            next(reader) # Skip header
            for row in reader:
                video_id, start_s, end_s, label = row
                # AVA labels: LAUGHTER, CLEAR_SPEECH, MUSIC, NOISE, OTHER
                is_laugh = 1 if label == "LAUGHTER" else 0
                start_s, end_s = float(start_s), float(end_s)
                duration = end_s - start_s

                # Construct expected clip paths (assuming download script saved them this way)
                # Example clip name format: {video_id}_{start_s:.3f}_{end_s:.3f}.wav/mp4
                # Need to confirm exact format from download_ava_laughter_clips.py if possible
                # For now, assume a simple structure - adjust if needed
                base_filename = f"{video_id}_{start_s:.3f}_{end_s:.3f}" # Placeholder format
                rel_audio_path = Path("ava/raw/clips") / f"{base_filename}.wav" # Relative to dataset_root
                rel_video_path = Path("ava/raw/clips") / f"{base_filename}.mp4" # Relative to dataset_root

                # Check if files actually exist relative to the dataset_root
                abs_audio_path = dataset_root / rel_audio_path
                abs_video_path = dataset_root / rel_video_path

                if not abs_audio_path.exists():
                    logging.debug(f"Skipping AVA entry: Audio file missing {abs_audio_path}")
                    continue
                if not abs_video_path.exists(): # AVA should have video
                    logging.debug(f"Skipping AVA entry: Video file missing {abs_video_path}")
                    continue

                # Generate 1-second windows
                num_windows = int(duration // TARGET_DURATION_S)
                for i in range(num_windows):
                    win_start = start_s + i * TARGET_DURATION_S
                    win_end = win_start + TARGET_DURATION_S
                    manifest_entries.append({
                        "rel_audio": str(rel_audio_path),
                        "rel_video": str(rel_video_path), # Include video path
                        "start": win_start, # Original start relative to full video
                        "end": win_end,     # Original end relative to full video
                        "label": is_laugh,
                        "source": "ava",
                        "has_video": 1 # Mark AVA as having video
                    })
    except Exception as e:
        logging.error(f"Error parsing AVA annotations: {e}")

    logging.info(f"Found {len(manifest_entries)} 1s windows from AVA.")
    return manifest_entries

def parse_ted(ted_dir, dataset_root):
    """Parses TED-Laughter labels."""
    logging.info(f"Parsing TED dataset from: {ted_dir}")
    ted_root = Path(ted_dir)
    dataset_root = Path(dataset_root)
    labels_file = ted_root / "labels.csv"
    audio_dir = ted_root / "audio"
    manifest_entries = []

    if not labels_file.exists():
        logging.warning(f"TED labels file not found: {labels_file}. Skipping TED.")
        return []
    if not audio_dir.is_dir():
        logging.warning(f"TED audio directory not found: {audio_dir}. Skipping TED.")
        return []

    try:
        df = pd.read_csv(labels_file)
        # Expected columns: file, start, end, label (1=laugh, 0=other)
        required_cols = {'file', 'start', 'end', 'label'}
        if not required_cols.issubset(df.columns):
             logging.error(f"TED labels file {labels_file} missing required columns. Found: {df.columns}")
             return []

        for _, row in df.iterrows():
            filename = row['file']
            start_s, end_s = float(row['start']), float(row['end'])
            label = int(row['label'])
            duration = end_s - start_s

            rel_audio_path = Path("ted/raw/audio") / filename # Relative to dataset_root
            abs_audio_path = dataset_root / rel_audio_path

            if not abs_audio_path.exists():
                logging.debug(f"Skipping TED entry: Audio file missing {abs_audio_path}")
                continue

            # TED is audio-only, generate 1s windows
            # The labels.csv should already contain 1s windows based on the plan
            if abs(duration - TARGET_DURATION_S) < 0.1: # Allow slight tolerance
                 manifest_entries.append({
                    "rel_audio": str(rel_audio_path),
                    "rel_video": "", # No video for TED
                    "start": start_s, # These are already window starts/ends
                    "end": end_s,
                    "label": label,
                    "source": "ted",
                    "has_video": 0 # Mark TED as audio-only
                })
            else:
                logging.debug(f"Skipping TED entry {filename}: Duration {duration:.2f}s is not close to {TARGET_DURATION_S}s.")

    except Exception as e:
        logging.error(f"Error parsing TED labels: {e}")

    logging.info(f"Found {len(manifest_entries)} 1s windows from TED.")
    return manifest_entries

def parse_voxceleb_negatives(vox_dir, dataset_root):
    """Parses placeholder VoxCeleb2 negative segments."""
    logging.info(f"Parsing VoxCeleb2 negatives from: {vox_dir}")
    vox_root = Path(vox_dir)
    dataset_root = Path(dataset_root)
    segments_dir = vox_root / "segments"
    manifest_entries = []

    if not segments_dir.is_dir():
        logging.warning(f"VoxCeleb2 segments directory not found: {segments_dir}. Skipping VoxCeleb negatives.")
        return []

    try:
        # Assuming segment filenames directly represent the audio files
        for segment_file in segments_dir.glob("*.wav"): # Adjust glob if format differs
            rel_audio_path = Path("voxceleb2_negatives/segments") / segment_file.name # Relative to dataset_root
            abs_audio_path = dataset_root / rel_audio_path

            if not abs_audio_path.exists():
                 logging.debug(f"Skipping VoxCeleb entry: Segment file missing {abs_audio_path}")
                 continue

            # VoxCeleb is audio-only, label as non-laugh (0)
            manifest_entries.append({
                "rel_audio": str(rel_audio_path),
                "rel_video": "", # No video
                "start": 0.0,    # Start of the segment itself
                "end": TARGET_DURATION_S, # Assuming segments are already 1s
                "label": 0,      # Negative sample
                "source": "voxceleb",
                "has_video": 0 # Mark as audio-only
            })
    except Exception as e:
        logging.error(f"Error parsing VoxCeleb2 negatives: {e}")

    logging.info(f"Found {len(manifest_entries)} 1s negative windows from VoxCeleb2.")
    return manifest_entries


def parse_smile(smile_dir, dataset_root):
    """Parses SMILE dataset segments and labels based on GT_laughter_reason.json."""
    logging.info(f"Parsing SMILE dataset from: {smile_dir}")
    smile_root = Path(smile_dir) # e.g., /home/ubuntu/datasets/SMILE
    dataset_root = Path(dataset_root) # e.g., /home/ubuntu/datasets
    # Corrected paths based on ls output
    anno_file = smile_root / "raw/SMILE_DATASET/annotations" / "GT_laughter_reason.json"
    video_segment_dir_abs = smile_root / "raw/SMILE_DATASET/videos" / "video_segments"
    audio_segment_dir_abs = smile_root / "raw/audio_segments" # Audio is directly under raw, not SMILE_DATASET

    # Relative paths for the manifest file (relative to dataset_root)
    audio_segment_dir_rel = Path("SMILE/raw/audio_segments")
    video_segment_dir_rel = Path("SMILE/raw/SMILE_DATASET/videos/video_segments")

    manifest_entries = []

    if not anno_file.exists():
        logging.warning(f"SMILE annotation file not found: {anno_file}. Skipping SMILE.")
        return []
    if not video_segment_dir_abs.is_dir():
         logging.warning(f"SMILE video segments directory not found: {video_segment_dir_abs}. Skipping SMILE.")
         return []
    if not audio_segment_dir_abs.is_dir():
         logging.warning(f"SMILE audio segments directory not found: {audio_segment_dir_abs}. Skipping SMILE.")
         return []

    try:
        with open(anno_file, 'r') as f:
            laughter_reasons = json.load(f)
        positive_segment_ids = set(laughter_reasons.keys())
        logging.info(f"Loaded {len(positive_segment_ids)} positive segment IDs from SMILE annotations.")

        all_video_segment_files = list(video_segment_dir_abs.glob("*.mp4")) # Find all video segments
        logging.info(f"Found {len(all_video_segment_files)} video segment files in {video_segment_dir_abs}.")

        for video_path_abs in tqdm(all_video_segment_files, desc="Processing SMILE Segments"):
            segment_filename = video_path_abs.name
            audio_filename = video_path_abs.with_suffix(".wav").name

            # --- Check file existence ---
            audio_path_abs = audio_segment_dir_abs / audio_filename
            if not video_path_abs.exists(): # Should always exist as we are iterating over them
                logging.warning(f"Video file listed by glob but not found: {video_path_abs}. Skipping.")
                continue
            if not audio_path_abs.exists():
                logging.debug(f"Skipping SMILE entry: Corresponding audio file missing {audio_path_abs}")
                continue

            # --- Extract metadata ---
            # Extract base ID (e.g., "1_10004" from "1_10004_seg_0.mp4")
            parts = segment_filename.split('_')
            if len(parts) < 2:
                logging.warning(f"Could not parse segment ID from filename: {segment_filename}. Skipping.")
                continue
            base_id = f"{parts[0]}_{parts[1]}"

            label = 1 if base_id in positive_segment_ids else 0

            # --- Construct relative paths for manifest ---
            rel_video_path = video_segment_dir_rel / segment_filename
            rel_audio_path = audio_segment_dir_rel / audio_filename

            # Placeholder start/end - dataloader will handle full segment loading/cropping
            start_s = 0.0
            end_s = 0.0 # Dataloader needs to get actual duration or use full clip

            manifest_entries.append({
                "rel_audio": str(rel_audio_path),
                "rel_video": str(rel_video_path),
                "start": start_s,
                "end": end_s,
                "label": label,
                "source": "smile",
                "has_video": 1 # Mark SMILE as having video
            })

    except Exception as e:
        logging.error(f"Error parsing SMILE dataset: {e}", exc_info=True)

    logging.info(f"Generated {len(manifest_entries)} manifest entries from SMILE.")
    return manifest_entries


def balance_manifest(manifest_df):
    """Balances the manifest to have 50% laugh and 50% non-laugh."""
    laughs = manifest_df[manifest_df['label'] == 1]
    non_laughs = manifest_df[manifest_df['label'] == 0]
    logging.info(f"Balancing: {len(laughs)} laughs vs {len(non_laughs)} non-laughs.")

    min_count = min(len(laughs), len(non_laughs))
    if min_count == 0:
        logging.warning("Cannot balance manifest: one class has zero samples.")
        return manifest_df # Return unbalanced if one class is empty

    balanced_df = pd.concat([
        laughs.sample(n=min_count, random_state=42),
        non_laughs.sample(n=min_count, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle

    logging.info(f"Balanced manifest size: {len(balanced_df)} ({min_count} per class).")
    return balanced_df

def main():
    parser = argparse.ArgumentParser(description="Prepare Humor Manifest CSV")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the final manifest CSV.")
    parser.add_argument("--dataset_root", type=str, required=True, help="Absolute path to the root directory containing all dataset subdirectories (e.g., /home/ubuntu/datasets).")
    parser.add_argument("--ava_dir", type=str, help="Subdirectory name for AVA dataset relative to dataset_root (e.g., ava).")
    parser.add_argument("--ted_dir", type=str, help="Subdirectory name for TED dataset relative to dataset_root (e.g., ted).")
    parser.add_argument("--crowd_dir", type=str, help="Subdirectory name for Crowd-Humour dataset (optional).")
    parser.add_argument("--vox_dir", type=str, help="Subdirectory name for VoxCeleb2 negatives dataset relative to dataset_root (e.g., voxceleb2_negatives).")
    parser.add_argument("--smile_dir", type=str, help="Subdirectory name for SMILE dataset relative to dataset_root (e.g., smile).") # Added SMILE arg
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_dir():
        logging.error(f"Dataset root directory not found: {dataset_root}")
        return

    all_entries = []

    # --- Parse Datasets ---
    if args.ava_dir:
        ava_full_path = dataset_root / args.ava_dir
        if ava_full_path.is_dir():
            all_entries.extend(parse_ava(ava_full_path, dataset_root))
        else:
            logging.warning(f"AVA directory not found: {ava_full_path}. Skipping.")
    if args.ted_dir:
        ted_full_path = dataset_root / args.ted_dir
        if ted_full_path.is_dir():
            all_entries.extend(parse_ted(ted_full_path, dataset_root))
        else:
            logging.warning(f"TED directory not found: {ted_full_path}. Skipping.")
    # Skipping Crowd-Humour as per previous steps
    # if args.crowd_dir:
    #     logging.info("Crowd-Humour parsing not implemented/skipped.")
    if args.vox_dir:
        vox_full_path = dataset_root / args.vox_dir
        if vox_full_path.is_dir():
             all_entries.extend(parse_voxceleb_negatives(vox_full_path, dataset_root))
        else:
            logging.warning(f"VoxCeleb directory not found: {vox_full_path}. Skipping.")
    if args.smile_dir: # Added SMILE call
        smile_full_path = dataset_root / args.smile_dir
        if smile_full_path.is_dir():
            all_entries.extend(parse_smile(smile_full_path, dataset_root))
        else:
            logging.warning(f"SMILE directory not found: {smile_full_path}. Skipping.")


    if not all_entries:
        logging.error("No valid, existing entries found from any dataset. Manifest cannot be created.")
        return

    # --- Combine and Balance ---
    manifest_df = pd.DataFrame(all_entries)
    logging.info(f"Total combined entries before balancing: {len(manifest_df)}")
    balanced_df = balance_manifest(manifest_df)

    # --- Save Manifest ---
    output_path = Path(args.output_dir) / "humor_manifest.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    balanced_df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    logging.info(f"Balanced humor manifest saved to: {output_path}")

if __name__ == "__main__":
    main()
