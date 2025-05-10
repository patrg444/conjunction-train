#!/usr/bin/env python
# Extract and cache multimodal embeddings (text, audio, video) for humor detection
# This script extracts embeddings from three models:
# 1. Text - XLM-RoBERta v3 embeddings (1024-D)
# 2. Audio - WavLM embeddings (768-D) - Updated from HuBERT
# 3. Video - Smile probability or vector from ResNet18 (1-D or 512-D)

import os
import argparse
import pandas as pd
import numpy as np
import torch
import yaml
from tqdm import tqdm
import logging
from pathlib import Path
import sys
import time
# Import Wav2Vec2Model for audio (WavLM is based on Wav2Vec2 architecture)
from transformers import XLMRobertaTokenizer, XLMRobertaModel, Wav2Vec2Model, Wav2Vec2Processor
import torchaudio
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import shutil # Import shutil for cleanup
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor as Pool
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_text_model(hf_model_path):
    """Load XLM-RoBERta model from a Hugging Face directory."""
    logger.info(f"Loading XLM-RoBERta from Hugging Face directory: {hf_model_path}")
    model = XLMRobertaModel.from_pretrained(hf_model_path)
    tokenizer = XLMRobertaTokenizer.from_pretrained(hf_model_path)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        logger.info("XLM-RoBERTa moved to CUDA")
    return model, tokenizer

def load_audio_model(model_dir_path):
    """Load WavLM model for audio embedding extraction from Hugging Face directory."""
    logger.info(f"Loading WavLM model from Hugging Face directory: {model_dir_path}")
    model = Wav2Vec2Model.from_pretrained(model_dir_path)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        logger.info("WavLM model moved to CUDA")
    return model

def load_video_model(ckpt_path):
    """Load ResNet18 smile classifier from Lightning checkpoint."""
    logger.info(f"Loading Smile ResNet18 from {ckpt_path}")
    model = resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1) # Match the checkpoint's final layer (smile logit)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    state = {k.replace('model.', ''): v for k, v in state.items()}
    missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
    if missing_keys:
        logger.warning(f"Missing keys when loading video model state_dict: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys when loading video model state_dict: {unexpected_keys}")
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        logger.info("Smile model moved to CUDA")
    return model

def extract_text_embedding(text, model, tokenizer, max_length=512):
    """Extract embedding from text using XLM-RoBERTa, skipping invalid or OOV."""
    if not isinstance(text, str) or not text.strip():
        return None
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    )
    if torch.cuda.is_available():
        inputs = {key: val.cuda() for key, val in inputs.items()}
    embed_size = model.get_input_embeddings().weight.size(0)
    ids = inputs["input_ids"].to(torch.long).clamp_(min=0, max=embed_size - 1)
    if ids.max() >= embed_size:
         logger.warning(f"Token ID {ids.max().item()} >= vocab size {embed_size} after clamp for text: '{text[:50]}...'")
         return None
    inputs["input_ids"] = ids
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        return embedding
    except Exception as e:
        logger.error(f"Error on text embedding for text '{text[:50]}...': {e}")
        return None

def extract_audio_embedding(audio_path, model, sample_rate=16000, max_duration=10):
    """Extract embedding from audio using WavLM."""
    try:
        waveform, orig_sr = torchaudio.load(audio_path)
        if orig_sr != sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=sample_rate)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        max_samples = max_duration * sample_rate
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
        elif waveform.shape[1] < max_samples:
            padding = max_samples - waveform.shape[1]
            waveform = F.pad(waveform, (0, padding))
        input_values = waveform.to(torch.float32)
        if input_values.dim() == 2:
            input_values = input_values.squeeze(0).unsqueeze(0)
        elif input_values.dim() == 3:
            input_values = input_values.squeeze(1)
        if torch.cuda.is_available():
            input_values = input_values.cuda()
        with torch.no_grad():
            outputs = model(input_values)
            embedding = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()[0]
        return embedding
    except Exception as e:
        logger.error(f"Error processing audio {audio_path}: {str(e)}")
        hidden_size = getattr(getattr(model, 'config', None), 'hidden_size', 768)
        return np.zeros(hidden_size, dtype=np.float32)

def extract_frames_ffmpeg(video_file, fps=4):
    """Extract frames from video_file at the given fps, return list of temp file paths."""
    tmp_dir = tempfile.mkdtemp()
    out_pattern = os.path.join(tmp_dir, "frame_%05d.jpg")
    cmd = ["ffmpeg", "-y", "-i", video_file, "-vf", f"fps={fps}", "-q:v", "2", out_pattern]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=60)
        frame_files = sorted([os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.endswith(".jpg")])
        return frame_files, tmp_dir
    except subprocess.TimeoutExpired:
        logger.error(f"ffmpeg timed out for {video_file}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return [], None
    except Exception as e:
        logger.error(f"ffmpeg failed for {video_file}: {e}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return [], None

def extract_video_embedding(
    video_path,
    model,
    video_fps=4,
    video_workers=4,
    save_frames_path=None,
    batch_size=32 # Added batch size for frame processing
):
    """
    Extract 512-D feature vector from video using ResNet18 by averaging frame features, processing frames in batches.
    If save_frames_path is provided, save per-frame probabilities to .npz.
    """
    tmp_dir = None
    try:
        if os.path.isdir(video_path):
            frame_files = sorted([os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        elif os.path.isfile(video_path) and video_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            frame_files, tmp_dir = extract_frames_ffmpeg(video_path, fps=video_fps)
        else:
            logger.warning(f"Video file processing not implemented or invalid path: {video_path}")
            return np.zeros(512, dtype=np.float32)

        if not frame_files:
            logger.warning(f"No frames extracted or found in {video_path}")
            if tmp_dir is not None: shutil.rmtree(tmp_dir, ignore_errors=True)
            return np.zeros(512, dtype=np.float32)

        transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        def load_and_transform(path):
            try:
                img = Image.open(path).convert('RGB')
                return transform(img)
            except Exception as e:
                logger.error(f"Failed to load/transform frame {path}: {e}")
                return None

        all_features = []
        all_probs = []
        original_fc = model.fc
        model.fc = nn.Identity() # Get features before final layer

        with Pool(max_workers=video_workers) as executor:
            # Process frames in batches to manage memory
            for i in range(0, len(frame_files), batch_size):
                batch_paths = frame_files[i:i+batch_size]
                images = [img for img in executor.map(load_and_transform, batch_paths) if img is not None]

                if not images:
                    logger.warning(f"No images loaded for batch {i//batch_size} of video {video_path}")
                    continue

                images_tensor = torch.stack(images)
                if torch.cuda.is_available():
                    images_tensor = images_tensor.cuda()

                with torch.no_grad():
                    features = model(images_tensor) # Shape [batch_size, 512]
                    all_features.append(features.cpu())
                    if save_frames_path is not None:
                        # Calculate probs for this batch if saving per-frame probs
                        model.fc = original_fc # Temporarily restore fc
                        prob_outputs = model.fc(features)
                        probs = torch.sigmoid(prob_outputs).cpu().numpy().flatten()
                        all_probs.extend(probs)
                        model.fc = nn.Identity() # Set back to identity

                # Clear CUDA cache periodically if needed, though batching is primary fix
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        model.fc = original_fc # Restore original FC layer finally

        if not all_features:
             logger.warning(f"No features extracted for video {video_path}")
             if tmp_dir is not None: shutil.rmtree(tmp_dir, ignore_errors=True)
             return np.zeros(512, dtype=np.float32)

        # Concatenate features from all batches and average
        all_features_tensor = torch.cat(all_features, dim=0)
        averaged_features = torch.mean(all_features_tensor, dim=0).numpy() # Shape [512]

        if save_frames_path is not None and all_probs:
             save_dict = {"probs": np.array(all_probs)}
             np.savez(save_frames_path, **save_dict)
             logger.info(f"Saved per-frame probabilities to {save_frames_path}")

        if tmp_dir is not None: shutil.rmtree(tmp_dir, ignore_errors=True)
        return averaged_features

    except Exception as e:
        logger.error(f"Error processing video {video_path}: {str(e)}")
        if tmp_dir is not None: shutil.rmtree(tmp_dir, ignore_errors=True)
        return np.zeros(512, dtype=np.float32)


def process_batch(
    manifest_df, config, models, embedding_dir, modalities=None, overwrite=False,
    video_save_frames=False, video_frames_dir=None, video_fps=4, video_workers=4,
    video_batch_size=32 # Added video frame batch size
):
    """Process a batch of data for embedding extraction."""
    text_model, text_tokenizer = models.get('text', (None, None))
    audio_model = models.get('audio', None)
    video_model = models.get('video', None)
    if modalities is None: modalities = ['text', 'audio', 'video']

    TEXT_DIM, AUDIO_DIM, VIDEO_DIM = 1024, 768, 512
    processed_count, skipped_count = 0, 0
    successful_saves = {'text': 0, 'audio': 0, 'video': 0}

    for idx, row in tqdm(manifest_df.iterrows(), total=len(manifest_df)):
        talk_id = row.get('talk_id')
        if not isinstance(talk_id, str) or not talk_id.startswith('urfunny_'):
            skipped_count += 1
            continue
        clip_id = talk_id

        # Text
        if 'text' in modalities and text_model:
            text_output_path = os.path.join(embedding_dir, 'text', f"{clip_id}.npy")
            if not os.path.exists(text_output_path) or overwrite:
                if 'text' in row and isinstance(row['text'], str) and row['text'].strip():
                    text_embedding = extract_text_embedding(row['text'], text_model, text_tokenizer)
                    if text_embedding is not None:
                        if not (np.isnan(text_embedding).any() or np.isinf(text_embedding).any()) and text_embedding.shape == (TEXT_DIM,):
                            os.makedirs(os.path.dirname(text_output_path), exist_ok=True)
                            np.save(text_output_path, text_embedding)
                            successful_saves['text'] += 1
                        else: logger.warning(f"Invalid text embedding for {clip_id}. Shape: {text_embedding.shape}, NaN/Inf: {np.isnan(text_embedding).any() or np.isinf(text_embedding).any()}")
                    else: logger.warning(f"Text embedding extraction failed for {clip_id}.")
                else: logger.warning(f"Missing/invalid text for {clip_id}.")

        # Audio
        if 'audio' in modalities and audio_model:
            audio_output_path = os.path.join(embedding_dir, 'audio', f"{clip_id}.npy")
            if not os.path.exists(audio_output_path) or overwrite:
                # --- FIX: Use 'rel_audio', ensure it's a non-empty string, and construct full path ---
                # Check if column exists, value is string (forced by dtype), and non-empty after stripping
                if 'rel_audio' in row and isinstance(row['rel_audio'], str) and row['rel_audio'].strip():
                    audio_filename = row['rel_audio'].strip() # Strip whitespace
                    # Assuming the script runs from /home/ubuntu/conjunction-train
                    audio_base_dir = "/home/ubuntu/conjunction-train/datasets/humor_datasets/ur_funny/audio/"
                    audio_path = os.path.join(audio_base_dir, audio_filename)
                    # --- END FIX ---
                    if os.path.exists(audio_path):
                        audio_embedding = extract_audio_embedding(audio_path, audio_model)
                        if audio_embedding is not None:
                            if not (np.isnan(audio_embedding).any() or np.isinf(audio_embedding).any()) and audio_embedding.shape == (AUDIO_DIM,):
                                os.makedirs(os.path.dirname(audio_output_path), exist_ok=True)
                                np.save(audio_output_path, audio_embedding)
                                successful_saves['audio'] += 1
                            else: logger.warning(f"Invalid audio embedding for {clip_id}. Shape: {audio_embedding.shape}, NaN/Inf: {np.isnan(audio_embedding).any() or np.isinf(audio_embedding).any()}")
                        else: logger.warning(f"Audio embedding extraction failed for {clip_id}.")
                    else: logger.warning(f"Constructed audio file path not found for {clip_id}: {audio_path}.")
                else:
                    # Log more specific info if possible
                    if 'rel_audio' not in row:
                        logger.warning(f"'rel_audio' column missing for {clip_id}.")
                    elif not isinstance(row.get('rel_audio'), str):
                         logger.warning(f"'rel_audio' value is not a string for {clip_id}: {type(row.get('rel_audio'))}")
                    elif not row.get('rel_audio', '').strip():
                         logger.warning(f"'rel_audio' value is empty or whitespace for {clip_id}.")
                    else: # Should not happen with the new check, but good fallback
                         logger.warning(f"Invalid/missing 'rel_audio' column for {clip_id} (Reason unknown). Value: {row.get('rel_audio')}")

        # Video
        if 'video' in modalities and video_model:
            video_output_path = os.path.join(embedding_dir, 'video', f"{clip_id}.npy")
            if not os.path.exists(video_output_path) or overwrite:
                if 'video_path' in row and pd.notna(row['video_path']) and isinstance(row['video_path'], str):
                    video_path = row['video_path']
                    if os.path.exists(video_path):
                        save_frames_path = None
                        if video_save_frames and video_frames_dir:
                            os.makedirs(video_frames_dir, exist_ok=True)
                            save_frames_path = os.path.join(video_frames_dir, f"{clip_id}.npz")
                        video_embedding = extract_video_embedding(
                            video_path, video_model, video_fps=video_fps,
                            video_workers=video_workers, save_frames_path=save_frames_path,
                            batch_size=video_batch_size # Pass batch size
                        )
                        if video_embedding is not None:
                            if not (np.isnan(video_embedding).any() or np.isinf(video_embedding).any()) and video_embedding.shape == (VIDEO_DIM,):
                                os.makedirs(os.path.dirname(video_output_path), exist_ok=True)
                                np.save(video_output_path, video_embedding)
                                successful_saves['video'] += 1
                            else: logger.warning(f"Invalid video embedding for {clip_id}. Shape: {video_embedding.shape}, NaN/Inf: {np.isnan(video_embedding).any() or np.isinf(video_embedding).any()}")
                        else: logger.warning(f"Video embedding extraction failed for {clip_id}.")
                    else: logger.warning(f"Video file not found for {clip_id}: {video_path}.")
                else: logger.warning(f"Invalid/missing video path for {clip_id}.")

    logger.info(f"Finished processing loop. Skipped {skipped_count} invalid manifest rows.")
    logger.info(f"Successfully saved embeddings: Text={successful_saves['text']}, Audio={successful_saves['audio']}, Video={successful_saves['video']}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract multimodal embeddings for humor detection')
    parser.add_argument('--manifest', type=str, required=True, help='Path to the manifest CSV file')
    parser.add_argument('--config', type=str, default='configs/model_checkpoint_paths.yaml', help='Path to the model config YAML')
    parser.add_argument('--embedding_dir', type=str, default='embeddings', help='Directory to save extracted embeddings')
    parser.add_argument('--modalities', type=str, nargs='+', choices=['text', 'audio', 'video'], default=['text', 'audio', 'video'], help='Which modalities to extract')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing embeddings')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='all', help='Which split to process')
    # parser.add_argument('--batch_size', type=int, default=64, help='Batch size for processing (currently unused)') # Removed as loop is row-by-row
    parser.add_argument('--video_save_frames', action='store_true', help='Save per-frame smile probabilities to video_frames/clipid.npz')
    parser.add_argument('--video_frames_dir', type=str, default=None, help='Directory to save per-frame video .npz files (default: embeddings/urfunny/video_frames)')
    parser.add_argument('--video_fps', type=int, default=4, help='FPS for video frame sampling (default: 4)')
    parser.add_argument('--video_workers', type=int, default=4, help='Number of parallel workers for frame decoding (default: 4)')
    parser.add_argument('--video_batch_size', type=int, default=32, help='Batch size for processing video frames (default: 32)') # Added video frame batch size arg
    # parser.add_argument('--video_save_features', action='store_true', help='Save penultimate features in .npz (not implemented)') # Removed as feature saving is internal detail
    return parser.parse_args()

def main():
    """Extract embeddings from multimodal models."""
    args = parse_args()
    config = load_config(args.config)
    for modality in args.modalities: os.makedirs(os.path.join(args.embedding_dir, modality), exist_ok=True)

    try:
        # Explicitly set dtype for rel_audio to avoid type inference issues
        manifest_df = pd.read_csv(args.manifest, dtype={'rel_audio': str})
        logger.info(f"Loaded manifest with {len(manifest_df)} entries")
    except Exception as e:
        logger.error(f"Failed to load manifest file {args.manifest}: {e}"); sys.exit(1)

    if args.split != 'all':
        if 'split' not in manifest_df.columns: logger.error(f"Manifest file {args.manifest} needs 'split' column for filtering."); sys.exit(1)
        manifest_df = manifest_df[manifest_df['split'] == args.split].reset_index(drop=True)
        logger.info(f"Filtered to {len(manifest_df)} entries for {args.split} split")

    models = {}
    ckpts = config['checkpoints']
    text_model_path = ckpts.get('xlm_roberta_v3', {}).get('model_path')
    audio_model_dir_path = ckpts.get('wavlm_laughter', {}).get('model_path')
    video_ckpt_path = ckpts.get('smile_resnet18', {}).get('model_path')

    if 'text' in args.modalities:
        # --- DEBUG PRINT ADDED ---
        logger.info(f"DEBUG: Attempting to load text model from path: {text_model_path}")
        # --- END DEBUG PRINT ---
        if text_model_path and os.path.isdir(text_model_path):
            models['text'] = load_text_model(text_model_path)
        else:
            logger.error(f"Text model path invalid: {text_model_path}"); sys.exit(1)
    if 'audio' in args.modalities:
        if audio_model_dir_path and os.path.isdir(audio_model_dir_path):
            models['audio'] = load_audio_model(audio_model_dir_path)
        else:
            logger.error(f"Audio model path invalid: {audio_model_dir_path}"); sys.exit(1)
    if 'video' in args.modalities:
        if video_ckpt_path and os.path.isfile(video_ckpt_path):
            models['video'] = load_video_model(video_ckpt_path)
        else:
            logger.error(f"Video model path invalid: {video_ckpt_path}"); sys.exit(1)

    video_frames_dir = args.video_frames_dir
    if args.video_save_frames and video_frames_dir is None:
        video_frames_dir = os.path.join(args.embedding_dir, "video_frames")
        logger.info(f"Defaulting video frames directory to: {video_frames_dir}")

    start_time = time.time()
    process_batch(
        manifest_df, config, models, args.embedding_dir, args.modalities, args.overwrite,
        args.video_save_frames, video_frames_dir, args.video_fps, args.video_workers,
        args.video_batch_size # Pass video batch size
    )
    elapsed_time = time.time() - start_time
    logger.info(f"Processing completed in {elapsed_time:.2f} seconds")

    logger.info("--- Embedding Extraction Summary ---")
    total_expected = len(manifest_df)
    for modality in args.modalities:
        modality_dir = os.path.join(args.embedding_dir, modality)
        if os.path.exists(modality_dir):
            num_embeddings = len([f for f in os.listdir(modality_dir) if f.endswith('.npy')])
            logger.info(f"Extracted {num_embeddings} / ~{total_expected} {modality} embeddings")
        else: logger.info(f"No embeddings directory found for {modality} at {modality_dir}")
    logger.info("------------------------------------")

if __name__ == '__main__':
    main()
