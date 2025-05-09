#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch Dataset class for loading humor detection data (laugh/non-laugh).
Handles mixed multimodal (audio+video) and audio-only samples.
Loads multiple labels if available in the manifest for multi-task learning.
"""

import os
import pandas as pd
import torch
import torch.nn as nn # Added missing import
import torch.nn.functional as F # Import functional for padding
from torch.utils.data import Dataset
import random # For SMILE random cropping
import logging # For warnings
import torchaudio
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from transformers import AutoFeatureExtractor, AutoTokenizer # Added AutoTokenizer

# Configure logging if not already configured elsewhere
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Media Loading Functions (Adapted from train_fusion_model.py and general practice) ---

def get_media_duration(path):
    """Gets the duration of an audio or video file in seconds."""
    try:
        # Try audio first
        info = torchaudio.info(path)
        return info.num_frames / info.sample_rate
    except Exception as audio_err:
        try:
            # Try video if audio fails
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise IOError(f"Cannot open video file: {path}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if fps > 0 and frame_count > 0:
                return frame_count / fps
            else:
                 raise ValueError(f"Invalid video metadata for {path} (fps={fps}, frames={frame_count})")
        except Exception as video_err:
            logging.error(f"Failed to get duration for {path}. Audio error: {audio_err}. Video error: {video_err}")
            return 0.0 # Return 0 duration if unable to determine

def load_audio_segment(path, start_sec, end_sec, target_sr, target_samples, feature_extractor):
    """Loads an audio segment, resamples, and processes with feature extractor."""
    try:
        waveform, sample_rate = torchaudio.load(path)

        # Resample if necessary
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
            waveform = resampler(waveform)
            sample_rate = target_sr # Update sample rate after resampling

        # Calculate start and end frames based on resampled audio
        start_frame = int(start_sec * sample_rate)
        end_frame = int(end_sec * sample_rate)
        num_frames = waveform.shape[1]

        # Ensure start/end frames are within bounds
        start_frame = max(0, start_frame)
        end_frame = min(num_frames, end_frame)

        # Extract the segment
        # Handle case where start/end indicate full file (e.g., from SMILE random crop)
        if start_sec == 0.0 and end_sec == 0.0: # Indicator for full file load before cropping
             segment = waveform.squeeze(0) # Use full waveform
        elif start_frame >= end_frame: # Handle edge case or very short segments
             logging.warning(f"Audio segment start >= end ({start_frame} >= {end_frame}) for {path}. Using small slice.")
             segment = waveform[:, start_frame:start_frame + 1].squeeze(0) # Take a tiny slice
        else:
             segment = waveform[:, start_frame:end_frame].squeeze(0) # Extract segment, remove channel dim

        # Process with feature extractor (handles padding/truncation)
        processed_audio = feature_extractor(
            segment,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding="longest",         # Pad to longest sequence in batch
            truncation=False,          # Don't truncate, rely on padding
            return_attention_mask=True # Explicitly request attention mask
        )
        audio_input_values = processed_audio.input_values.squeeze(0) # Remove batch dim
        # Directly access attention mask, relying on the outer try-except to catch errors
        audio_attention_mask = processed_audio.attention_mask.squeeze(0)

        # --- Enforce fixed length ---
        max_audio_len = 250000 # Define a fixed max length (e.g., ~15.6s at 16kHz)
        current_len = audio_input_values.shape[0]

        if current_len < max_audio_len:
            # Pad
            pad_len = max_audio_len - current_len
            audio_input_values = F.pad(audio_input_values, (0, pad_len), "constant", 0)
            audio_attention_mask = F.pad(audio_attention_mask, (0, pad_len), "constant", 0) # Pad mask with 0s
        elif current_len > max_audio_len:
            # Truncate
            audio_input_values = audio_input_values[:max_audio_len]
            audio_attention_mask = audio_attention_mask[:max_audio_len]

        # Ensure shapes are consistent after padding/truncation
        if audio_input_values.shape[0] != max_audio_len or audio_attention_mask.shape[0] != max_audio_len:
             logging.error(f"Shape mismatch after padding/truncation for {path}. Expected {max_audio_len}, got input: {audio_input_values.shape}, mask: {audio_attention_mask.shape}")
             # Return None to skip sample if something went wrong
             return None, None

        return audio_input_values, audio_attention_mask

    except (AttributeError, KeyError, Exception) as e: # Catch errors during loading or mask access
        # Downgrade to warning and return None to signal failure
        logging.warning(f"Error loading/processing audio {path} ({start_sec}-{end_sec}s): {e}. Skipping sample.")
        return None, None # Signal failure


def load_video_segment(path, start_sec, end_sec, target_fps, target_frames, img_size, transform, temporal_augmentations, is_training):
    """Loads a video segment, samples frames, applies transforms."""
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if video_fps <= 0 or total_frames <= 0:
            raise ValueError(f"Invalid video metadata for {path} (fps={video_fps}, frames={total_frames})")

        # Calculate start and end frames in the original video
        start_frame_orig = int(start_sec * video_fps)
        end_frame_orig = int(end_sec * video_fps)

        # Handle case where start/end indicate full file (e.g., from SMILE random crop)
        if start_sec == 0.0 and end_sec == 0.0:
            start_frame_orig = 0
            end_frame_orig = total_frames

        # Ensure start/end frames are within bounds
        start_frame_orig = max(0, start_frame_orig)
        end_frame_orig = min(total_frames, end_frame_orig)
        segment_frame_count = end_frame_orig - start_frame_orig

        if segment_frame_count <= 0:
             raise ValueError(f"Calculated segment frame count is zero or negative for {path} ({start_frame_orig} -> {end_frame_orig})")

        # Sample target_frames from the segment
        # Use center sampling for validation/test, random sampling for train
        if is_training:
            # Randomly sample start frame within the segment
            max_start_offset = max(0, segment_frame_count - target_frames)
            start_offset = random.randint(0, max_start_offset)
            effective_start_frame = start_frame_orig + start_offset
        else:
            # Center crop within the segment
            start_offset = max(0, (segment_frame_count - target_frames) // 2)
            effective_start_frame = start_frame_orig + start_offset

        effective_end_frame = min(end_frame_orig, effective_start_frame + target_frames)

        # Ensure we sample exactly target_frames, handling segments shorter than target
        indices = np.linspace(effective_start_frame, effective_end_frame - 1, target_frames, dtype=np.int32)
        # Clip indices to be within the valid range of the original video
        indices = np.clip(indices, 0, total_frames - 1)

        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                if frames: # Repeat last frame if read fails
                    frames.append(frames[-1].copy())
                    logging.warning(f"Repeating last frame for index {i} in video {path}")
                else: # If first frame fails, use black frame
                    frames.append(np.zeros((img_size, img_size, 3), dtype=np.uint8))
                    logging.warning(f"Using black frame for index {i} in video {path}")
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (img_size, img_size))
            frames.append(frame)
        cap.release()

        # Apply per-frame transforms
        # Ensure transform is not None before applying
        if transform:
            transformed_frames = [transform(Image.fromarray(f)) for f in frames]
        else:
            # Fallback if no transform provided (e.g., convert to tensor directly)
            transformed_frames = [transforms.ToTensor()(Image.fromarray(f)) for f in frames]

        video_tensor = torch.stack(transformed_frames) # [T, C, H, W]

        # Apply temporal augmentations
        if is_training and temporal_augmentations:
             video_tensor = temporal_augmentations(video_tensor)

        return video_tensor

    except Exception as e:
        logging.error(f"Error loading/processing video {path} ({start_sec}-{end_sec}s): {e}", exc_info=True)
        # Return zeros matching expected shape
        return torch.zeros(target_frames, 3, img_size, img_size) # T, C, H, W


class HumorDataset(Dataset):
    """
    Dataset for humor detection, loading audio and optionally video.
    Returns samples with audio, video (or zeros if audio-only), text features, and multiple labels if available.
    """
    def __init__(self, manifest_path, dataset_root, duration=1.0, sample_rate=16000, video_fps=15, video_frames=None, img_size=112, hubert_model_name="facebook/hubert-base-ls960", text_model_name="distilbert-base-uncased", max_text_len=128, split='train', augment=True):
        """
        Args:
            manifest_path (str): Path to the balanced humor manifest CSV.
            dataset_root (str): Root directory where dataset files are stored.
            duration (float): Duration of segments to load in seconds.
            sample_rate (int): Target audio sample rate.
            video_fps (int): Target video frames per second.
            video_frames (int, optional): Fixed number of frames for video. If None, calculated from duration*fps.
            img_size (int): Target size for video frames (height and width).
            hubert_model_name (str): Name of the Hubert model for audio feature extraction.
            text_model_name (str): Name of the model for text tokenization.
            max_text_len (int): Maximum sequence length for text tokenizer.
            split (str): Dataset split ('train', 'val', 'test'). Controls augmentation.
            augment (bool): Whether to apply augmentations (typically True for train split).
        """
        logging.info(f"Initializing HumorDataset for split '{split}' from: {manifest_path}")
        try:
            # Load full manifest first if split column exists
            df_full = pd.read_csv(manifest_path)
            if 'split' in df_full.columns:
                 self.manifest = df_full[df_full['split'] == split].reset_index(drop=True)
                 logging.info(f"Filtered manifest for split '{split}': {len(self.manifest)} rows")
            else:
                 # If no split column, assume the manifest is already for the intended split
                 self.manifest = df_full
                 logging.warning(f"Manifest {manifest_path} has no 'split' column. Using all {len(self.manifest)} rows for split '{split}'.")

            # Basic validation - check for required columns
            required_cols = ['rel_audio', 'rel_video', 'start', 'end', 'label', 'source', 'has_video', 'transcript']
            if not all(col in self.manifest.columns for col in required_cols):
                 missing = [col for col in required_cols if col not in self.manifest.columns]
                 raise ValueError(f"Manifest missing required columns for HumorDataset: {missing}. Found: {list(self.manifest.columns)}")

            # Check for optional multi-task label columns
            self.optional_label_cols = ['emotion_label', 'humor_label', 'smile_label', 'joke_label']
            self.available_label_cols = [col for col in self.optional_label_cols if col in self.manifest.columns]
            if len(self.available_label_cols) < len(self.optional_label_cols):
                missing_optional = [col for col in self.optional_label_cols if col not in self.manifest.columns]
                logging.warning(f"Manifest is missing optional label columns: {missing_optional}. These tasks will be skipped during training/evaluation if their labels are needed.")

        except Exception as e:
            logging.error(f"Could not load or parse manifest {manifest_path}: {e}", exc_info=True)
            raise FileNotFoundError(f"Could not load or parse manifest {manifest_path}: {e}")

        self.dataset_root = dataset_root
        self.duration = duration
        self.sample_rate = sample_rate # Target SR for audio loading
        self.video_fps = video_fps
        self.video_frames = video_frames if video_frames else int(duration * video_fps)
        self.img_size = img_size
        self.split = split
        self.augment = augment and split == 'train' # Only augment training split
        self.max_text_len = max_text_len

        # --- Initialize Hubert Feature Extractor ---
        logging.info(f"Initializing Hubert Feature Extractor for {hubert_model_name}...")
        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(hubert_model_name)
            # Ensure target_sr matches feature extractor's expectation
            if self.sample_rate != self.feature_extractor.sampling_rate:
                 logging.warning(f"Dataset sample rate ({self.sample_rate}) differs from Hubert feature extractor ({self.feature_extractor.sampling_rate}). Audio will be resampled.")
                 self.sample_rate = self.feature_extractor.sampling_rate # Use extractor's SR
            logging.info(f"Target sampling rate set to: {self.sample_rate}")
        except Exception as e:
            logging.error(f"Failed to load Hubert feature extractor '{hubert_model_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to load Hubert feature extractor '{hubert_model_name}': {e}")

        # --- Initialize Text Tokenizer ---
        logging.info(f"Initializing Text Tokenizer for {text_model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
            logging.info(f"Text tokenizer loaded. Max length: {self.max_text_len}")
        except Exception as e:
            logging.error(f"Failed to load text tokenizer '{text_model_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to load text tokenizer '{text_model_name}': {e}")


        # --- Setup Video Transforms (borrowed from train_fusion_model.py) ---
        # TODO: Consider making augmentations configurable via a config dict
        self.video_transform, self.temporal_augmentations = self._get_video_transforms(augment=self.augment)

        logging.info(f"HumorDataset initialized for split '{split}' with {len(self.manifest)} samples.")
        logging.info(f"  Target Duration: {self.duration}s, SR: {self.sample_rate}, FPS: {self.video_fps}")
        logging.info(f"  Target Video Frames: {self.video_frames}, Image Size: {self.img_size}")
        logging.info(f"  Target Text Length: {self.max_text_len}")
        logging.info(f"  Augmentation enabled: {self.augment}")
        logging.info(f"  Available optional labels: {self.available_label_cols}")


    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        """Loads a single sample (audio features, video frames, text, labels)."""
        try: # Wrap entire item loading in try-except
            row = self.manifest.iloc[idx]
            source = row['source']
            rel_audio_path = row['rel_audio']
            rel_video_path = row['rel_video']
            manifest_start = row['start'] # Note: For SMILE, this is 0.0 from prepare_script
            manifest_end = row['end']     # Note: For SMILE, this is 0.0 from prepare_script
            has_video_flag = row['has_video'] == 1
            transcript = str(row['transcript']) if pd.notna(row['transcript']) else "" # Load transcript, handle NaN

            audio_path = os.path.join(self.dataset_root, rel_audio_path)
            video_path = os.path.join(self.dataset_root, rel_video_path) if has_video_flag and pd.notna(rel_video_path) and rel_video_path != '' else None

            # Determine if this is a SMILE segment needing random cropping
            is_smile_segment = (manifest_start == 0.0 and manifest_end == 0.0 and source == 'smile')

            segment_start_s = manifest_start
            segment_end_s = manifest_end
            target_duration = self.duration

            if is_smile_segment:
                # --- Load Full SMILE Segment and Random Crop ---
                actual_duration = get_media_duration(audio_path)

                if actual_duration <= 0:
                    logging.error(f"Could not get valid duration for SMILE segment: {audio_path}. Skipping sample.")
                    return None # Skip sample if duration is invalid
                elif actual_duration < target_duration:
                    logging.warning(f"SMILE segment {rel_audio_path} duration ({actual_duration:.2f}s) is less than target ({target_duration}s). Using full segment.")
                    segment_start_s = 0.0
                    segment_end_s = 0.0 # Use 0/0 as indicator to load_audio/video to load full file
                else:
                    max_start = actual_duration - target_duration
                    crop_start_s = random.uniform(0.0, max_start)
                    segment_start_s = crop_start_s
                    segment_end_s = crop_start_s + target_duration

                # Load the potentially cropped segment
                audio_input_values, audio_attention_mask = load_audio_segment(
                    audio_path, segment_start_s, segment_end_s, self.sample_rate, int(self.sample_rate * target_duration), self.feature_extractor
                )
                if audio_input_values is None: return None # Skip if audio failed

                if video_path:
                    video_tensor = load_video_segment(
                        video_path, segment_start_s, segment_end_s, self.video_fps, self.video_frames, self.img_size, self.video_transform, self.temporal_augmentations, self.augment
                    )
                else:
                    video_tensor = torch.zeros(self.video_frames, 3, self.img_size, self.img_size) # TCHW

            else:
                # --- Load AVA/TED/VoxCeleb Segment (already target duration window) ---
                audio_input_values, audio_attention_mask = load_audio_segment(
                     audio_path, manifest_start, manifest_end, self.sample_rate, int(self.sample_rate * target_duration), self.feature_extractor
                )
                if audio_input_values is None: return None # Skip if audio failed

                if has_video_flag and video_path:
                    video_tensor = load_video_segment(
                        video_path, manifest_start, manifest_end, self.video_fps, self.video_frames, self.img_size, self.video_transform, self.temporal_augmentations, self.augment
                    )
                else:
                    video_tensor = torch.zeros(self.video_frames, 3, self.img_size, self.img_size) # TCHW

            # --- Labels ---
            # Load primary laugh label (from 'label' column)
            laugh_label = torch.tensor(int(row['label']), dtype=torch.long)

            # Load optional labels if columns exist, handle missing values (NaN -> None)
            sample_labels = {'laugh_label': laugh_label}
            for col in self.available_label_cols:
                label_val = row.get(col)
                if pd.isna(label_val):
                    sample_labels[col] = None # Represent missing label as None
                else:
                    # Convert to appropriate type (int for classification, float for regression/binary?)
                    # Assuming all are classification/binary for now
                    try:
                        # Use float for binary tasks expecting BCEWithLogitsLoss later
                        if col in ['humor_label', 'smile_label', 'joke_label']:
                             sample_labels[col] = torch.tensor(float(label_val), dtype=torch.float)
                        else: # Assume int/long for multi-class like emotion
                             sample_labels[col] = torch.tensor(int(label_val), dtype=torch.long)
                    except ValueError:
                         logging.warning(f"Could not convert label '{label_val}' for column '{col}' in row {idx}. Setting to None.")
                         sample_labels[col] = None # Set to None if conversion fails

            # --- Tokenize Text ---
            tokenized_text = self.tokenizer(
                transcript,
                padding='max_length',    # Pad to max_length
                truncation=True,         # Truncate to max_length
                max_length=self.max_text_len,
                return_tensors='pt',     # Return PyTorch tensors
                return_attention_mask=True
            )
            text_input_ids = tokenized_text['input_ids'].squeeze(0) # Remove batch dim
            text_attention_mask = tokenized_text['attention_mask'].squeeze(0) # Remove batch dim


            # --- Return Sample ---
            sample = {
                'audio_input_values': audio_input_values,
                'audio_attention_mask': audio_attention_mask,
                'video': video_tensor,
                'text_input_ids': text_input_ids,
                'text_attention_mask': text_attention_mask,
                'has_video': torch.tensor(has_video_flag and video_path is not None, dtype=torch.bool),
                'source': source
            }
            # Add all loaded labels to the sample dictionary
            sample.update(sample_labels)

            return sample

        except Exception as e:
            logging.error(f"Error processing sample at index {idx}: {e}", exc_info=True)
            return None # Return None to be filtered by collate_fn


    def _get_video_transforms(self, augment):
        """Create video augmentation pipeline (similar to train_fusion_model.py)."""
        # Basic transforms (applied per frame)
        transform_list = [
            transforms.ToTensor(), # Converts PIL Image [H, W, C] to tensor [C, H, W]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet norm
        ]
        temporal_augmentations = []

        if augment:
            # Add augmentations here if needed, e.g., RandomHorizontalFlip, ColorJitter
            # Example: transform_list.insert(0, transforms.RandomHorizontalFlip(p=0.5))
            # Example temporal: temporal_augmentations.append(RandomTimeReverse(p=0.5))
            pass # Keep it simple for now, add augmentations based on config later if needed

        return transforms.Compose(transform_list), nn.Sequential(*temporal_augmentations)


# --- Example Usage (for testing this script directly) ---
if __name__ == '__main__':
    print("--- Testing HumorDataset ---")

    # Create dummy files and manifest for a more realistic test
    DUMMY_ROOT = "dummy_humor_data"
    DUMMY_MANIFEST = os.path.join(DUMMY_ROOT, "dummy_humor_manifest.csv")
    os.makedirs(os.path.join(DUMMY_ROOT, "smile/raw/audio_segments"), exist_ok=True)
    os.makedirs(os.path.join(DUMMY_ROOT, "smile/raw/videos/video_segments"), exist_ok=True)
    os.makedirs(os.path.join(DUMMY_ROOT, "ted/raw/audio"), exist_ok=True)

    # Dummy SMILE files (need actual audio/video for loading)
    # Create short dummy wav and mp4 files
    SAMPLE_RATE = 16000
    torch.manual_seed(0)
    dummy_audio_smile = torch.randn(1, SAMPLE_RATE * 3) # 3 seconds
    dummy_audio_ted = torch.randn(1, SAMPLE_RATE * 1)   # 1 second
    torchaudio.save(os.path.join(DUMMY_ROOT, "smile/raw/audio_segments/smile_1.wav"), dummy_audio_smile, SAMPLE_RATE)
    torchaudio.save(os.path.join(DUMMY_ROOT, "ted/raw/audio/ted_1.wav"), dummy_audio_ted, SAMPLE_RATE)
    # Dummy video (create a few black frames) - requires opencv-python
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        dummy_video_path = os.path.join(DUMMY_ROOT, "smile/raw/videos/video_segments/smile_1.mp4")
        out = cv2.VideoWriter(dummy_video_path, fourcc, 15.0, (64, 64))
        for _ in range(45): # 3 seconds at 15 fps
            out.write(np.zeros((64, 64, 3), dtype=np.uint8))
        out.release()
        print(f"Created dummy video: {dummy_video_path}")
    except Exception as e:
        print(f"Could not create dummy video (opencv-python needed?): {e}")


    dummy_data = {
        'rel_audio': ["smile/raw/audio_segments/smile_1.wav", "ted/raw/audio/ted_1.wav", "smile/raw/audio_segments/smile_1.wav"],
        'rel_video': ["smile/raw/videos/video_segments/smile_1.mp4", "", "smile/raw/videos/video_segments/smile_1.mp4"],
        'start': [0.0, 0.0, 0.0], # SMILE uses 0/0, TED uses actual start (0.0 for this dummy)
        'end': [0.0, 1.0, 0.0],   # SMILE uses 0/0, TED uses actual end (1.0 for this dummy)
        'label': [1, 0, 1], # Primary laugh label
        'source': ['smile', 'ted', 'smile'],
        'has_video': [1, 0, 1],
        'split': ['train', 'train', 'val'], # Add split column
        'transcript': ["This is funny.", "Not funny.", "Maybe funny?"],
        # Add optional labels (some missing)
        'emotion_label': [3, 4, np.nan], # Happiness, Sadness, Missing
        'humor_label': [1.0, 0.0, 1.0],
        'smile_label': [1.0, 0.0, np.nan],
        'joke_label': [0.0, 0.0, 1.0]
    }
    pd.DataFrame(dummy_data).to_csv(DUMMY_MANIFEST, index=False)
    print(f"Created dummy manifest: {DUMMY_MANIFEST}")

    try:
        # Test dataset initialization
        dataset_train = HumorDataset(
            manifest_path=DUMMY_MANIFEST,
            dataset_root=DUMMY_ROOT,
            split='train',
            augment=False # Disable augmentation for simple test
        )
        print(f"\nTrain Dataset size: {len(dataset_train)}")

        if len(dataset_train) > 0:
            print("\nLoading train sample 0 (SMILE)...")
            sample0 = dataset_train[0]
            if sample0: # Check if sample loading succeeded
                print(" Sample 0 keys:", sample0.keys())
                print(" Audio Input Values shape:", sample0['audio_input_values'].shape) # Processed by Hubert extractor
                print(" Audio Attention Mask shape:", sample0['audio_attention_mask'].shape)
                print(" Video shape:", sample0['video'].shape)
                print(" Text Input IDs shape:", sample0['text_input_ids'].shape)
                print(" Text Attention Mask shape:", sample0['text_attention_mask'].shape)
                print(" Laugh Label:", sample0.get('laugh_label'))
                print(" Emotion Label:", sample0.get('emotion_label'))
                print(" Humor Label:", sample0.get('humor_label'))
                print(" Smile Label:", sample0.get('smile_label'))
                print(" Joke Label:", sample0.get('joke_label'))
                print(" Has Video:", sample0['has_video'])
                print(" Source:", sample0['source'])
            else:
                print(" Sample 0 failed to load.")

            print("\nLoading train sample 1 (TED)...")
            sample1 = dataset_train[1]
            if sample1:
                print(" Sample 1 keys:", sample1.keys())
                print(" Audio Input Values shape:", sample1['audio_input_values'].shape)
                print(" Audio Attention Mask shape:", sample1['audio_attention_mask'].shape)
                print(" Video shape:", sample1['video'].shape) # Should be zeros
                print(" Text Input IDs shape:", sample1['text_input_ids'].shape)
                print(" Text Attention Mask shape:", sample1['text_attention_mask'].shape)
                print(" Laugh Label:", sample1.get('laugh_label'))
                print(" Emotion Label:", sample1.get('emotion_label'))
                print(" Humor Label:", sample1.get('humor_label'))
                print(" Smile Label:", sample1.get('smile_label'))
                print(" Joke Label:", sample1.get('joke_label'))
                print(" Has Video:", sample1['has_video']) # Should be False
                print(" Source:", sample1['source'])
            else:
                print(" Sample 1 failed to load.")

        # Test validation split
        dataset_val = HumorDataset(
            manifest_path=DUMMY_MANIFEST,
            dataset_root=DUMMY_ROOT,
            split='val',
            augment=False
        )
        print(f"\nValidation Dataset size: {len(dataset_val)}")
        if len(dataset_val) > 0:
            print("\nLoading val sample 0 (SMILE)...")
            sample_val = dataset_val[0]
            if sample_val:
                print(" Val Sample keys:", sample_val.keys())
                print(" Laugh Label:", sample_val.get('laugh_label'))
                print(" Emotion Label:", sample_val.get('emotion_label')) # Should be None due to NaN
                print(" Humor Label:", sample_val.get('humor_label'))
                print(" Smile Label:", sample_val.get('smile_label')) # Should be None due to NaN
                print(" Joke Label:", sample_val.get('joke_label'))
            else:
                print(" Val Sample 0 failed to load.")


    except Exception as e:
        print(f"\nError during dataset test: {e}")
        logging.error("Dataset test failed", exc_info=True)

    finally:
        # Clean up dummy files/dirs
        import shutil
        if os.path.exists(DUMMY_ROOT):
            shutil.rmtree(DUMMY_ROOT)
            print(f"\nCleaned up dummy data directory: {DUMMY_ROOT}")

    print("\n--- HumorDataset Test Finished ---")
