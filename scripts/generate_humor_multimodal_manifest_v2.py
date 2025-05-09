#!/usr/bin/env python
import os
import argparse
import pandas as pd
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

def generate_multimodal_manifest(
    base_manifest_path,
    audio_embedding_dir,
    text_embedding_dir,
    video_au_embedding_dir,
    output_manifest_path,
    id_col='talk_id',
    label_col='label',
    split_col='split' # Assuming 'split' column exists in the base manifest
):
    logger.info(f"Loading base manifest from: {base_manifest_path}")
    try:
        base_df = pd.read_csv(base_manifest_path)
    except Exception as e:
        logger.error(f"Failed to load base manifest: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"Base manifest loaded with {len(base_df)} entries.")
    
    output_data = []
    missing_audio_count = 0
    missing_text_count = 0
    missing_video_au_count = 0

    for idx, row in tqdm(base_df.iterrows(), total=len(base_df), desc="Generating Multimodal Manifest"):
        clip_id = str(row[id_col])
        label = row[label_col]
        split = row.get(split_col, 'train') # Default to 'train' if split column is missing

        audio_emb_path = os.path.join(audio_embedding_dir, f"{clip_id}.npy")
        text_emb_path = os.path.join(text_embedding_dir, f"{clip_id}.npy")
        video_au_emb_path = os.path.join(video_au_embedding_dir, f"{clip_id}.npy")

        # Check for existence of embedding files
        audio_exists = os.path.exists(audio_emb_path)
        text_exists = os.path.exists(text_emb_path)
        video_au_exists = os.path.exists(video_au_emb_path)

        if not audio_exists:
            missing_audio_count += 1
            logger.debug(f"Audio embedding not found for {clip_id} at {audio_emb_path}")
        if not text_exists:
            missing_text_count += 1
            logger.debug(f"Text embedding not found for {clip_id} at {text_emb_path}")
        if not video_au_exists:
            missing_video_au_count += 1
            logger.debug(f"Video AU embedding not found for {clip_id} at {video_au_emb_path}")
            # Even if missing, we might still want to include the path to the placeholder (empty .npy)
            # The dataloader will handle empty arrays.

        output_data.append({
            id_col: clip_id,
            label_col: label,
            split_col: split,
            'sequential_audio_path': audio_emb_path if audio_exists else None,
            'pooled_full_text_path': text_emb_path if text_exists else None,
            'sequential_video_au_path': video_au_emb_path # Always include path, Dataloader handles missing/empty
        })

    if missing_audio_count > 0:
        logger.warning(f"Total missing audio embeddings: {missing_audio_count}/{len(base_df)}")
    if missing_text_count > 0:
        logger.warning(f"Total missing text embeddings: {missing_text_count}/{len(base_df)}")
    if missing_video_au_count > 0:
        logger.warning(f"Total missing/placeholder video AU embeddings: {missing_video_au_count}/{len(base_df)}")

    output_df = pd.DataFrame(output_data)
    
    # Filter out rows where essential embeddings are missing (e.g., audio or text)
    # For video, we allow it to be missing (will be handled by dataloader as zeros)
    initial_count = len(output_df)
    output_df.dropna(subset=['sequential_audio_path', 'pooled_full_text_path'], inplace=True)
    filtered_count = len(output_df)
    if initial_count > filtered_count:
        logger.info(f"Filtered out {initial_count - filtered_count} rows due to missing essential audio or text embeddings.")

    logger.info(f"Saving new multimodal manifest to: {output_manifest_path} with {len(output_df)} entries.")
    output_df.to_csv(output_manifest_path, index=False)
    logger.info("Multimodal manifest generation complete.")

def main():
    parser = argparse.ArgumentParser(description='Generate a new multimodal manifest for UR-FUNNY humor detection.')
    parser.add_argument('--base_manifest_path', type=str, required=True,
                        help='Path to the base manifest CSV (e.g., datasets/manifests/humor/urfunny_raw_data_complete.csv).')
    parser.add_argument('--audio_embedding_dir', type=str, required=True,
                        help='Directory containing sequential audio embeddings (e.g., datasets/humor_embeddings/audio_wavlm_base_plus_sequential/).')
    parser.add_argument('--text_embedding_dir', type=str, required=True,
                        help='Directory containing pooled full text embeddings (e.g., datasets/humor_embeddings_v2/text_pooled_full_xlmr/).')
    parser.add_argument('--video_au_embedding_dir', type=str, required=True,
                        help='Directory containing sequential video AU embeddings (e.g., datasets/humor_embeddings_v2/video_sequential_openface_au/).')
    parser.add_argument('--output_manifest_path', type=str, required=True,
                        help='Path to save the new multimodal manifest CSV (e.g., datasets/manifests/humor/urfunny_multimodal_v2_final.csv).')
    parser.add_argument('--id_col', type=str, default='talk_id', help='ID column name in base manifest.')
    parser.add_argument('--label_col', type=str, default='label', help='Label column name in base manifest.')
    parser.add_argument('--split_col', type=str, default='split', help='Split column name in base manifest.')

    args = parser.parse_args()

    generate_multimodal_manifest(
        args.base_manifest_path,
        args.audio_embedding_dir,
        args.text_embedding_dir,
        args.video_au_embedding_dir,
        args.output_manifest_path,
        id_col=args.id_col,
        label_col=args.label_col,
        split_col=args.split_col
    )

if __name__ == '__main__':
    main()
