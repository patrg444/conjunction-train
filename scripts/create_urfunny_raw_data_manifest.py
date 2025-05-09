#!/usr/bin/env python
import os
import argparse
import pandas as pd
import logging
from tqdm import tqdm
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def extract_numeric_id(talk_id_str, prefix="urfunny_"):
    """Extracts numeric part from talk_id like 'urfunny_123' -> '123'."""
    if isinstance(talk_id_str, str) and talk_id_str.startswith(prefix):
        return talk_id_str[len(prefix):]
    return str(talk_id_str) # Fallback if no prefix or not string

def main():
    parser = argparse.ArgumentParser(description='Create a comprehensive manifest with raw data paths for UR-FUNNY.')
    parser.add_argument('--text_manifest_path', type=str, required=True,
                        help='Path to the input text manifest CSV (e.g., ur_funny_train_humor_cleaned.csv).')
    parser.add_argument('--video_base_dir', type=str, required=True,
                        help='Base directory for raw video files on the target system (e.g., /home/ubuntu/conjunction-train/datasets/humor/urfunny/raw/urfunny2_videos/).')
    parser.add_argument('--audio_base_dir', type=str, required=True,
                        help='Base directory for raw audio files on the target system (e.g., /home/ubuntu/conjunction-train/datasets/ur_funny/audio/).')
    parser.add_argument('--output_manifest_path', type=str, required=True,
                        help='Path to save the new comprehensive manifest CSV.')
    parser.add_argument('--text_col', type=str, default='transcript',
                        help='Column name in text manifest for the full transcript.')
    parser.add_argument('--id_col', type=str, default='talk_id',
                        help='Column name in text manifest for unique IDs.')
    parser.add_argument('--label_col', type=str, default='label',
                        help='Column name in text manifest for labels.')
    parser.add_argument('--delimiter', type=str, default=' ||| ',
                        help='Delimiter for context and punchline in the transcript.')
    parser.add_argument('--split_name', type=str, required=True,
                        help='Name of the split (e.g., train, val, test). This will be added as a column.')
    parser.add_argument('--video_extension', type=str, default='.mp4', help="Extension for video files.")
    parser.add_argument('--audio_extension', type=str, default='.wav', help="Extension for audio files.")

    args = parser.parse_args()

    try:
        text_df = pd.read_csv(args.text_manifest_path)
        logger.info(f"Loaded text manifest: {args.text_manifest_path} with {len(text_df)} entries.")
    except Exception as e:
        logger.error(f"Failed to load text manifest file {args.text_manifest_path}: {e}", exc_info=True)
        sys.exit(1)

    output_rows = []
    for idx, row in tqdm(text_df.iterrows(), total=len(text_df), desc=f"Processing {args.split_name} split"):
        talk_id = str(row[args.id_col])
        transcript = str(row[args.text_col])
        label = row[args.label_col]

        context_text = ""
        punchline_text = ""
        if args.delimiter in transcript:
            parts = transcript.split(args.delimiter, 1)
            context_text = parts[0].strip()
            punchline_text = parts[1].strip()
        else:
            logger.warning(f"Delimiter '{args.delimiter}' not found for {talk_id}. Using full transcript as context.")
            context_text = transcript.strip()

        # Assuming talk_id in text manifest is like 'urfunny_123' and media files are '123.mp4'
        numeric_id = extract_numeric_id(talk_id)
        
        raw_video_path = os.path.join(args.video_base_dir, numeric_id + args.video_extension)
        raw_audio_path = os.path.join(args.audio_base_dir, numeric_id + args.audio_extension)

        output_rows.append({
            'talk_id': talk_id, # Keep original talk_id for reference
            'numeric_id': numeric_id,
            'raw_context_text': context_text,
            'raw_punchline_text': punchline_text,
            'raw_audio_path': raw_audio_path,
            'raw_video_path': raw_video_path,
            'label': label,
            'split': args.split_name
        })

    output_df = pd.DataFrame(output_rows)

    # Check if the output file already exists to handle headers correctly
    file_exists = os.path.isfile(args.output_manifest_path)
    
    try:
        os.makedirs(os.path.dirname(args.output_manifest_path), exist_ok=True)
        if not file_exists:
            # File doesn't exist, write with header
            output_df.to_csv(args.output_manifest_path, index=False, mode='w')
            logger.info(f"Successfully created new comprehensive raw data manifest: {args.output_manifest_path} with {len(output_df)} entries.")
        else:
            # File exists, append without header
            output_df.to_csv(args.output_manifest_path, index=False, mode='a', header=False)
            logger.info(f"Successfully appended {len(output_df)} entries to existing manifest: {args.output_manifest_path}")

    except Exception as e:
        logger.error(f"Failed to save or append to output manifest {args.output_manifest_path}: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    # Ensure the main output file is clean before starting the first split (e.g., train)
    # This part is tricky if run for multiple splits; typically, you'd handle cleanup outside
    # or ensure the first call creates it and subsequent calls append.
    # For this workflow, we'll assume the user manages the initial state of the output file
    # if running this script multiple times to build up a single manifest.
    # A safer approach for multi-split runs would be to output to temp files and then combine.
    # However, given the current issues, let's try making the script append-aware.
    
    # If it's the 'train' split, we can assume it's the first run and overwrite the output file.
    # For 'val' and 'test', it should append. This logic is now handled by mode='w'/'a'.
    main()
