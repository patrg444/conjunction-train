#!/usr/bin/env python
import os
import argparse
import pandas as pd
import numpy as np
import torch
import yaml
from tqdm import tqdm
import logging
from transformers import Wav2Vec2Model # WavLM uses Wav2Vec2 architecture
import torchaudio
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_audio_model(model_name_or_path):
    logger.info(f"Loading WavLM/Wav2Vec2 model from: {model_name_or_path}")
    model = Wav2Vec2Model.from_pretrained(model_name_or_path)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        logger.info("Audio model moved to CUDA")
    return model

def extract_sequential_audio_embedding(audio_path, model, target_sample_rate=16000, max_duration_sec=None):
    try:
        waveform, orig_sr = torchaudio.load(audio_path)
        if orig_sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sample_rate)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1: # Ensure mono
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if max_duration_sec is not None:
            max_samples = int(max_duration_sec * target_sample_rate)
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]
            # No padding here, we want the raw sequence length up to max_duration

        input_values = waveform.to(torch.float32) # Should be [1, num_samples]

        if torch.cuda.is_available():
            input_values = input_values.cuda()

        with torch.no_grad():
            outputs = model(input_values)
            # IMPORTANT: Save the full sequence, do not apply mean pooling here
            sequential_embeddings = outputs.last_hidden_state.squeeze(0).cpu().numpy()
            # Expected shape: (num_frames, feature_dim)
        return sequential_embeddings.astype(np.float32)
    except Exception as e:
        logger.error(f"Error processing audio {audio_path}: {str(e)}")
        # Determine expected feature_dim for error case
        feature_dim = getattr(getattr(model, 'config', None), 'hidden_size', 768) # Default to 768 if not found
        return np.zeros((1, feature_dim), dtype=np.float32) # Return a minimal sequence on error

def process_manifest_audios(
    manifest_df,
    audio_model,
    embedding_dir,
    audio_col='rel_audio', # Column in manifest with relative audio path
    audio_base_dir="/home/ubuntu/conjunction-train/datasets/humor_datasets/ur_funny/audio/", # Base path for relative audio files
    clip_id_col='talk_id',
    overwrite=False,
    max_duration_sec=None # Optional: to cap sequence length
):
    os.makedirs(embedding_dir, exist_ok=True)
    logger.info(f"Saving sequential audio embeddings to: {embedding_dir}")

    for idx, row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="Extracting Sequential Audio Embeddings"):
        clip_id = row[clip_id_col]
        output_filename = f"{clip_id}.npy"
        output_path = os.path.join(embedding_dir, output_filename)

        if os.path.exists(output_path) and not overwrite:
            # logger.info(f"Skipping {clip_id}, embedding already exists.")
            continue

        if audio_col not in row or not isinstance(row[audio_col], str) or not row[audio_col].strip():
            logger.warning(f"Missing or invalid audio path for {clip_id} in column '{audio_col}'. Skipping.")
            continue

        relative_audio_path = row[audio_col].strip()
        full_audio_path = os.path.join(audio_base_dir, relative_audio_path)

        if not os.path.exists(full_audio_path):
            logger.warning(f"Audio file not found for {clip_id} at {full_audio_path}. Skipping.")
            continue

        sequential_embedding = extract_sequential_audio_embedding(full_audio_path, audio_model, max_duration_sec=max_duration_sec)

        if sequential_embedding is not None and sequential_embedding.ndim == 2 and sequential_embedding.shape[0] > 0:
            np.save(output_path, sequential_embedding)
        else:
            logger.warning(f"Failed to extract valid sequential embedding for {clip_id}. Embedding shape: {sequential_embedding.shape if sequential_embedding is not None else 'None'}")

def main():
    parser = argparse.ArgumentParser(description='Extract sequential WavLM/Wav2Vec2 embeddings for humor detection audio.')
    parser.add_argument('--manifest', type=str, required=True, help='Path to the manifest CSV (must contain audio paths and clip IDs).')
    parser.add_argument('--audio_model_name_or_path', type=str, default='microsoft/wavlm-base-plus', help='Hugging Face model name or path to local model directory for WavLM/Wav2Vec2.')
    parser.add_argument('--embedding_dir', type=str, required=True, help='Directory to save extracted sequential audio embeddings.')
    parser.add_argument('--audio_col', type=str, default='rel_audio', help='Column name in manifest for relative audio file paths.')
    parser.add_argument('--audio_base_dir', type=str, default="/home/ubuntu/conjunction-train/datasets/humor_datasets/ur_funny/audio/", help='Base directory for relative audio paths.')
    parser.add_argument('--clip_id_col', type=str, default='talk_id', help='Column name in manifest for clip/talk IDs.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing embeddings.')
    parser.add_argument('--max_audio_duration', type=float, default=None, help='Optional: Maximum audio duration in seconds to process (trims longer audio).')

    args = parser.parse_args()

    try:
        manifest_df = pd.read_csv(args.manifest, dtype={args.audio_col: str, args.clip_id_col: str})
        logger.info(f"Loaded manifest: {args.manifest} with {len(manifest_df)} entries.")
    except Exception as e:
        logger.error(f"Failed to load manifest file {args.manifest}: {e}")
        sys.exit(1)

    audio_model = load_audio_model(args.audio_model_name_or_path)

    process_manifest_audios(
        manifest_df,
        audio_model,
        args.embedding_dir,
        audio_col=args.audio_col,
        audio_base_dir=args.audio_base_dir,
        clip_id_col=args.clip_id_col,
        overwrite=args.overwrite,
        max_duration_sec=args.max_audio_duration
    )
    logger.info("Sequential audio embedding extraction complete.")

if __name__ == '__main__':
    main()
