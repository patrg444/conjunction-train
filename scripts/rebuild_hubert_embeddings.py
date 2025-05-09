#!/usr/bin/env python3
"""
Rebuilds HuBERT embeddings NPZ files from audio files listed in split CSVs,
ensuring correct label alignment and including basenames.
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
import torchaudio
from transformers import HubertModel, Wav2Vec2FeatureExtractor # Use transformers
from pathlib import Path
from tqdm import tqdm
import hashlib

# --- Configuration ---
TARGET_SAMPLE_RATE = 16000

# Consistent emotion labels and mapping (must match training script)
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
LABEL_TO_IDX = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
IDX_TO_LABEL = {idx: label for label, idx in LABEL_TO_IDX.items()}

# --- Helper Functions ---

def get_basename(filepath):
    """Extracts the basename without extension."""
    return Path(filepath).stem

def load_and_resample_audio(audio_path, target_sr=TARGET_SAMPLE_RATE):
    """Loads audio, converts to mono, and resamples."""
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0] > 1: # Convert to mono by averaging channels
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
            waveform = resampler(waveform)
        # Return waveform suitable for processor (list of tensors/arrays)
        return waveform.squeeze(0).numpy() # Processor often prefers numpy
    except Exception as e:
        print(f"ERROR loading/resampling {audio_path}: {e}")
        return None

def get_file_sha256(filepath):
    """Computes SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(4096)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()

# --- Main Extraction Logic ---

def extract_embeddings_for_split(
    split_name,
    split_csv_path,
    manifest_df,
    audio_root_dir,
    output_dir,
    model,
    processor, # Add processor argument
    device,
    batch_size=16
):
    """Extracts embeddings for all files listed in a split CSV."""
    print(f"\n--- Processing Split: {split_name} ---")

    if not split_csv_path.exists():
        print(f"ERROR: Split CSV not found: {split_csv_path}. Skipping.")
        return False

    try:
        split_df = pd.read_csv(split_csv_path)
        # Determine the path column name ('path' or 'FileName')
        path_col_in_split = 'path' if 'path' in split_df.columns else 'FileName'
        if path_col_in_split not in split_df.columns:
            print(f"ERROR: Path column ('path' or 'FileName') not found in {split_csv_path}. Skipping.")
            return False
        print(f"Found {len(split_df)} items listed in {split_csv_path.name}")
    except Exception as e:
        print(f"ERROR: Failed to load split CSV {split_csv_path}: {e}. Skipping.")
        return False

    # Create mapping from basename to label using the main manifest
    # Assumes manifest_df already has 'basename' and 'label_idx' columns
    basename_to_label = manifest_df.set_index('basename')['label_idx'].to_dict()
    basename_to_path = manifest_df.set_index('basename')['path'].to_dict()

    all_embeddings = []
    all_labels = []
    all_basenames = []
    processed_count = 0
    error_count = 0

    # Process in batches for efficiency
    file_tuples = [] # Store (basename, full_audio_path, label)
    for _, row in split_df.iterrows():
        basename = get_basename(row[path_col_in_split])
        if basename not in basename_to_path or basename not in basename_to_label:
            print(f"WARNING: Basename '{basename}' from split CSV not found in manifest. Skipping.")
            error_count += 1
            continue

        relative_path = basename_to_path[basename]
        label_idx = basename_to_label[basename]

        # Construct full path
        if Path(relative_path).is_absolute():
            full_audio_path = Path(relative_path)
        else:
            if not audio_root_dir:
                 print(f"ERROR: Manifest path '{relative_path}' is relative, but --audio_root_dir was not provided. Skipping.")
                 error_count += 1
                 continue
            full_audio_path = Path(audio_root_dir) / relative_path

        if not full_audio_path.exists():
            print(f"WARNING: Audio file not found: {full_audio_path}. Skipping.")
            error_count += 1
            continue

        file_tuples.append((basename, full_audio_path, label_idx))

    print(f"Processing {len(file_tuples)} found audio files in batches...")
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        for i in tqdm(range(0, len(file_tuples), batch_size), desc=f"Extracting {split_name}"):
            batch_tuples = file_tuples[i : i + batch_size]
            waveforms_list = [] # List to hold numpy arrays for processor
            current_batch_labels = []
            current_batch_basenames = []

            # Load audio first
            for basename, path, label in batch_tuples:
                waveform_np = load_and_resample_audio(path) # Returns numpy array or None
                if waveform_np is not None:
                    waveforms_list.append(waveform_np)
                    current_batch_labels.append(label)
                    current_batch_basenames.append(basename)
                else:
                    error_count += 1

            if not waveforms_list: # Skip batch if all files failed to load
                continue

            # Use the processor to prepare the batch
            inputs = processor(
                waveforms_list,
                sampling_rate=TARGET_SAMPLE_RATE,
                return_tensors="pt",
                padding=True # Processor handles padding
            )

            try:
                # Move processed inputs to the target device
                input_values = inputs.input_values.to(device)
                attention_mask = inputs.attention_mask.to(device) # Use attention mask for padded sequences

                # Extract features using the transformers model
                outputs = model(input_values, attention_mask=attention_mask)

                # Get last hidden state (standard output for feature extraction)
                last_hidden_states = outputs.last_hidden_state # Shape: (batch, seq_len_model, features)

                # Determine the sequence lengths from the model output and the original attention mask
                seq_len_model = last_hidden_states.shape[1]
                seq_len_mask_orig = attention_mask.shape[1] # Original mask length from processor

                # Find the minimum common length to ensure alignment
                common_len = min(seq_len_model, seq_len_mask_orig)

                # Slice both the hidden states and the original attention mask to this common length
                last_hidden_states_synced = last_hidden_states[:, :common_len, :]
                attention_mask_synced = attention_mask[:, :common_len]

                # Mean pool across the time dimension, considering the synced attention mask
                # Mask out padding before pooling using the synced mask
                masked_hidden_states = last_hidden_states_synced * attention_mask_synced.unsqueeze(-1)
                summed_hidden_states = masked_hidden_states.sum(dim=1)

                # Calculate actual lengths based on the *synced* mask
                actual_lengths = attention_mask_synced.sum(dim=1).unsqueeze(-1).clamp(min=1)
                pooled_features = summed_hidden_states / actual_lengths # Shape: (batch_size, feature_dim)

                # Store results
                all_embeddings.extend(pooled_features.cpu().numpy()) # Move to CPU before converting to numpy
                all_labels.extend(current_batch_labels)
                all_basenames.extend(current_batch_basenames)
                processed_count += len(current_batch_basenames)

            except Exception as e:
                 print(f"\nERROR during batch {i//batch_size} feature extraction: {e}")
                 # Mark all items in this batch as errored
                 error_count += len(current_batch_basenames) # Count errors based on successfully loaded waveforms


    print(f"Finished split {split_name}. Processed: {processed_count}, Errors: {error_count}")

    if not all_embeddings:
        print("ERROR: No embeddings were extracted for this split. Skipping NPZ save.")
        return False

    # Save to NPZ
    output_npz_path = output_dir / f"{split_name}_embeddings_regen.npz"
    try:
        np.savez_compressed(
            output_npz_path,
            embeddings=np.array(all_embeddings, dtype=np.float32),
            labels=np.array(all_labels, dtype=np.int32),
            basenames=np.array(all_basenames, dtype='S') # Save basenames as bytes
        )
        print(f"Successfully saved: {output_npz_path}")
        # Calculate and print SHA256 hash
        sha256_hash = get_file_sha256(output_npz_path)
        print(f"  SHA256: {sha256_hash}")
        return True
    except Exception as e:
        print(f"ERROR saving NPZ file {output_npz_path}: {e}")
        return False


# --- Main Function ---

def main():
    parser = argparse.ArgumentParser(description="Rebuild HuBERT embeddings NPZ files.")
    parser.add_argument("--manifest", type=str, default="data/audio_manifest_remapped.tsv", help="Path to the corrected main audio manifest file (TSV)")
    parser.add_argument("--splits_dir", type=str, default="splits", help="Directory containing train/val/test/crema* split CSVs")
    parser.add_argument("--audio_root_dir", type=str, required=True, help="Root directory containing the raw audio files referenced in the manifest")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save the regenerated NPZ files")
    parser.add_argument("--model_name", type=str, default="facebook/hubert-large-ls960-ft", help="HuggingFace model name or path to local Fairseq checkpoint for HuBERT")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'cpu', or None for auto-detect)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for GPU processing")

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    splits_dir = Path(args.splits_dir)
    audio_root_dir = Path(args.audio_root_dir) # Keep as Path object
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Device Setup ---
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cpu':
        print("WARNING: Running on CPU, extraction will be slow.")

    # --- Load Model ---
    print(f"Loading model: {args.model_name}...")
    try:
        # Use HuggingFace transformers library
        processor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name)
        model = HubertModel.from_pretrained(args.model_name)
        model.to(device)
        print("Model and processor loaded successfully using transformers.")
    except Exception as e:
        print(f"ERROR: Failed to load model '{args.model_name}' using transformers. Ensure transformers is installed and model name is correct: {e}")
        return

    # --- Load Manifest ---
    print(f"Loading manifest: {manifest_path}")
    try:
        manifest_df = pd.read_csv(manifest_path, sep='\t', header=None, names=['path', 'label'])
        manifest_df['basename'] = manifest_df['path'].apply(get_basename)
        manifest_df['label_idx'] = manifest_df['label'].astype(int) # Assume labels are already integers
        print(f"Loaded {len(manifest_df)} items from manifest.")
    except Exception as e:
        print(f"ERROR: Failed to load or process manifest {manifest_path}: {e}")
        return

    # --- Process Splits ---
    splits_to_process = [
        "train", "val", "test", "crema_d_train", "crema_d_val"
    ]
    all_successful = True

    for split_name in splits_to_process:
        csv_filename = f"{split_name}.csv"
        split_csv_path = splits_dir / csv_filename
        success = extract_embeddings_for_split(
            split_name,
            split_csv_path,
            manifest_df,
            audio_root_dir, # Pass the validated Path object
            output_dir,
            model,
            processor, # Pass processor
            device,
            args.batch_size
        )
        if not success:
            all_successful = False

    print("\n--- Extraction Summary ---")
    if all_successful:
        print("All splits processed and NPZ files saved successfully.")
    else:
        print("One or more splits failed during processing or saving. Check logs above.")

if __name__ == "__main__":
    main()
