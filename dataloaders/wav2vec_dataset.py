#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wav2vec_dataset.py

PyTorch Dataset for loading audio data for wav2vec fine-tuning.
Reads a manifest file (path<TAB>label_id), loads audio (either
from .wav files or streams from video containers), resamples to 16kHz,
normalizes, and prepares input for the wav2vec model.
"""

import os
import subprocess
import io
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np
import traceback

# Target sample rate for wav2vec models
TARGET_SAMPLE_RATE = 16000
# Target RMS level for normalization (in dBFS)
TARGET_RMS_DBFS = -26.0

class Wav2VecDataset(Dataset):
    """
    PyTorch Dataset for loading audio waveforms for Wav2Vec2 fine-tuning.
    """
    def __init__(self, manifest_path: str, processor, max_duration_s: float = 10.0, min_duration_s: float = 0.5):
        """
        Args:
            manifest_path (str): Path to the manifest TSV file (path<TAB>label_id).
            processor: HuggingFace Wav2Vec2Processor instance.
            max_duration_s (float): Maximum audio duration in seconds to keep.
            min_duration_s (float): Minimum audio duration in seconds to keep.
        """
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

        self.processor = processor
        self.max_samples = int(max_duration_s * TARGET_SAMPLE_RATE)
        self.min_samples = int(min_duration_s * TARGET_SAMPLE_RATE)

        # Load manifest
        self.entries = pd.read_csv(self.manifest_path, sep='\t', header=None, names=['path', 'label_id'])
        print(f"Loaded manifest: {len(self.entries)} entries from {manifest_path}")

        # Pre-filter based on duration if possible (only works if --extract-wav was used)
        # This is an optimization to avoid loading very long/short files during training
        # If streaming from video, duration check happens during __getitem__
        initial_count = len(self.entries)
        valid_indices = []
        can_check_duration = all(Path(p).suffix.lower() == '.wav' for p in self.entries['path'])

        if can_check_duration:
            print("Pre-filtering based on duration (requires extracted .wav files)...")
            for idx, row in self.entries.iterrows():
                try:
                    info = sf.info(row['path'])
                    duration_ok = self.min_samples <= info.frames <= self.max_samples
                    if duration_ok:
                        valid_indices.append(idx)
                except Exception as e:
                    print(f"Warning: Could not get info for {row['path']}: {e}")
                    # Keep it for now, will fail in __getitem__ if problematic
                    valid_indices.append(idx)
            self.entries = self.entries.iloc[valid_indices].reset_index(drop=True)
            print(f"Filtered entries by duration: {initial_count} -> {len(self.entries)}")
        else:
             print("Cannot pre-filter by duration (manifest contains non-.wav files). Duration check will happen during loading.")


    def __len__(self):
        return len(self.entries)

    def _load_audio_wav(self, file_path: Path):
        """Loads audio from a .wav file."""
        try:
            waveform, sample_rate = torchaudio.load(file_path)
            # Convert to mono if necessary
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            # Resample if necessary
            if sample_rate != TARGET_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
                waveform = resampler(waveform)
            return waveform.squeeze(0) # Return 1D tensor
        except Exception as e:
            print(f"Error loading WAV {file_path}: {e}")
            print(traceback.format_exc())
            return None

    def _load_audio_stream(self, file_path: Path):
        """Streams audio from video container using ffmpeg."""
        try:
            cmd = [
                "ffmpeg",
                "-i", str(file_path),
                "-vn",                 # No video
                "-acodec", "pcm_s16le", # Signed 16-bit PCM
                "-ar", str(TARGET_SAMPLE_RATE), # Target sample rate
                "-ac", "1",            # Mono
                "-f", "wav",           # Output format WAV
                "-",                   # Output to stdout
                "-loglevel", "error"   # Suppress verbose output
            ]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                print(f"ffmpeg error for {file_path}: {stderr.decode()}")
                return None

            # Load waveform from the byte stream
            waveform, sample_rate = torchaudio.load(io.BytesIO(stdout))
            # Should already be mono and correct sample rate due to ffmpeg args
            return waveform.squeeze(0) # Return 1D tensor

        except FileNotFoundError:
             print("Error: ffmpeg command not found. Ensure ffmpeg is installed and in PATH.")
             raise
        except Exception as e:
            print(f"Error streaming audio from {file_path}: {e}")
            print(traceback.format_exc())
            return None

    def _normalize_rms(self, waveform: torch.Tensor, target_dbfs: float = TARGET_RMS_DBFS) -> torch.Tensor:
        """Normalize waveform to a target RMS level in dBFS."""
        if waveform.numel() == 0: return waveform # Handle empty tensor
        rms = waveform.pow(2).mean().sqrt()
        if rms < 1e-8: return waveform # Avoid division by zero / amplifying silence

        target_linear = 10**(target_dbfs / 20.0)
        gain = target_linear / rms
        return waveform * gain

    def __getitem__(self, idx):
        entry = self.entries.iloc[idx]
        file_path = Path(entry['path'])
        label_id = int(entry['label_id'])

        # Load audio based on file extension
        if file_path.suffix.lower() == '.wav':
            waveform = self._load_audio_wav(file_path)
        else:
            waveform = self._load_audio_stream(file_path)

        if waveform is None or waveform.numel() == 0:
            # Return None or raise error? For now, return None, collate_fn should handle it.
            print(f"Warning: Failed to load audio for index {idx}, path {file_path}. Skipping.")
            # Need to return something with the expected structure for the collator
            # Return dummy data that collate_fn can filter out
            return {"input_values": None, "attention_mask": None, "label": label_id, "valid": False}


        # Duration check (especially needed if streaming)
        if not (self.min_samples <= waveform.shape[0] <= self.max_samples):
             print(f"Warning: Skipping index {idx} due to duration {waveform.shape[0]/TARGET_SAMPLE_RATE:.2f}s (min={self.min_duration_s}, max={self.max_duration_s}) Path: {file_path}")
             return {"input_values": None, "attention_mask": None, "label": label_id, "valid": False}


        # Normalize
        waveform = self._normalize_rms(waveform)

        # Process with Wav2Vec2Processor
        # This handles padding/truncation based on the processor's settings
        processed = self.processor(
            waveform.numpy(), # Processor expects numpy array
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt", # Return PyTorch tensors
            padding="longest", # Pad to longest in batch (handled by collator usually, but good default)
            truncation=True,   # Truncate if longer than model max length
            max_length=self.max_samples # Ensure processor knows max length
        )

        # Squeeze the batch dimension added by the processor
        input_values = processed.input_values.squeeze(0)
        attention_mask = processed.attention_mask.squeeze(0)

        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "label": label_id,
            "valid": True # Mark as valid sample
        }


def collate_fn_pad(batch):
    """
    Custom collate function to handle padding and filter out invalid samples.
    Invalid samples are those where audio loading failed or duration was outside limits.
    """
    # Filter out invalid samples
    valid_batch = [item for item in batch if item["valid"]]

    if not valid_batch:
        # If the whole batch is invalid, return empty tensors with correct types/devices
        # This requires knowing the device, which is tricky here. Assume CPU for now.
        # The trainer might need specific handling for empty batches.
        return {
             'input_values': torch.empty((0, 0), dtype=torch.float),
             'attention_mask': torch.empty((0, 0), dtype=torch.long),
             'labels': torch.empty((0,), dtype=torch.long)
        }


    # Pad 'input_values' and 'attention_mask'
    input_features = [{"input_values": item["input_values"]} for item in valid_batch]
    # Use the processor's pad method for consistency
    # We need an instance of the processor here, which is awkward.
    # Alternative: Pad manually using torch.nn.utils.rnn.pad_sequence
    # Let's assume the processor is available globally or passed somehow (not ideal)
    # For now, pad manually:
    input_values_list = [item["input_values"] for item in valid_batch]
    attention_mask_list = [item["attention_mask"] for item in valid_batch]

    padded_input_values = torch.nn.utils.rnn.pad_sequence(input_values_list, batch_first=True, padding_value=0.0) # Wav2Vec processor uses 0.0 for padding
    padded_attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0) # Attention mask padding is 0

    # Collect labels
    labels = torch.tensor([item["label"] for item in valid_batch], dtype=torch.long)

    return {
        'input_values': padded_input_values,
        'attention_mask': padded_attention_mask,
        'labels': labels # Trainer expects 'labels' key
    }


# Example Usage (requires a manifest file and HuggingFace processor)
if __name__ == '__main__':
    from transformers import Wav2Vec2Processor

    # Create dummy manifest for testing
    dummy_manifest = "data/dummy_audio_manifest.tsv"
    dummy_json = "data/dummy_class_counts.json"
    os.makedirs("data", exist_ok=True)
    # Assume you have some audio files (e.g., from running build_audio_manifest --extract-wav)
    # Or point to existing video files if you want to test streaming
    # Example: Use existing spectrogram script path for testing path parsing
    test_file_path = "scripts/preprocess_spectrograms.py" # Just need a valid path string
    with open(dummy_manifest, "w") as f:
        f.write(f"{test_file_path}\t3\n") # Example: Happy
        f.write(f"{test_file_path}\t0\n") # Example: Angry
        f.write(f"{test_file_path}\t4\n") # Example: Neutral

    print(f"Created dummy manifest: {dummy_manifest}")

    # Initialize processor
    try:
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

        # Create dataset instance
        # NOTE: This will fail if test_file_path is not a valid audio/video file
        # Replace test_file_path with actual paths to test properly
        print("\nAttempting to create dataset (will likely fail with dummy path)...")
        try:
            dataset = Wav2VecDataset(dummy_manifest, processor)
            print(f"Dataset length: {len(dataset)}")

            # Get one sample (will fail if paths are invalid)
            # sample = dataset[0]
            # print("\nSample structure:")
            # print({k: v.shape if isinstance(v, torch.Tensor) else v for k, v in sample.items()})

            # Test DataLoader with collate function
            # from torch.utils.data import DataLoader
            # dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn_pad)
            # batch = next(iter(dataloader))
            # print("\nBatch structure:")
            # print({k: v.shape if isinstance(v, torch.Tensor) else v for k, v in batch.items()})

        except Exception as e:
            print(f"\nDataset/DataLoader test failed as expected with dummy paths: {e}")
            print("Replace dummy paths in manifest with real audio/video paths to test fully.")

    except OSError as e:
         print(f"Could not load Wav2Vec2Processor. Make sure you have internet connection and 'transformers' installed: {e}")

    # Clean up dummy files
    # os.remove(dummy_manifest)
    # os.remove(dummy_json) # build_audio_manifest creates this, not this script
    print(f"\n(Note: Dummy manifest '{dummy_manifest}' was left for inspection)")
