#!/usr/bin/env python3
"""
Multimodal Emotion Recognition Training Script (SlowFast + HuBERT Fusion)
"""

import os
import sys
import argparse
# Add project root to sys.path to allow finding sibling modules like ser_hubert
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import yaml
import math
from pathlib import Path
import cv2
from PIL import Image
import warnings # Import warnings
import traceback # Add this import

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler # Removed autocast import here, will use torch.amp.autocast
import torchvision
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau # Added CosineAnnealingWarmRestarts, ReduceLROnPlateau
import torchaudio # For audio loading
from collections import Counter # For class weights
# Explicitly set the backend to ffmpeg, hoping it's more robust - Removed this problematic line
# try:
#     torchaudio.set_audio_backend("ffmpeg")
#     print("Successfully set torchaudio backend to ffmpeg.")
# except RuntimeError as e:
#     print(f"Warning: Could not set torchaudio backend to ffmpeg: {e}")
#     print("Proceeding with default backend...")
print(f"Using torchaudio backend: {torchaudio.get_audio_backend()}") # Print the actual backend
from transformers import AutoFeatureExtractor # For audio processing

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from datetime import datetime

# Import the correct classifier from the script that generated the checkpoint
try:
    # Use absolute import assuming script is run from conjunction-train root
    from scripts.train_slowfast_emotion import EmotionClassifier as VideoEmotionClassifier
    print("Successfully imported EmotionClassifier from scripts.train_slowfast_emotion")
except ImportError as e:
    print(f"Error importing EmotionClassifier from scripts.train_slowfast_emotion: {e}")
    print("Ensure train_slowfast_emotion.py is in the 'scripts' directory relative to the execution path on EC2.")
    sys.exit(1)

# Import the Hubert SER model
try:
    # Assuming ser_hubert is adjacent or in python path
    from ser_hubert.hubert_ser_module import HubertSER
    print("Successfully imported HubertSER from ser_hubert.hubert_ser_module")
except ImportError as e:
    print(f"Error importing HubertSER: {e}")
    print("Ensure ser_hubert directory is accessible.")
    sys.exit(1)


# Emotion labels - ensure consistency
EMOTION_LABELS = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad'] # Make sure this matches SlowFast and HuBERT data


# --- Augmentation Classes (Copied from train_slowfast_emotion.py) ---
class RandomTimeReverse(nn.Module):
    """Randomly reverse video clip in temporal dimension."""
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    def forward(self, x):
        if random.random() < self.p:
            return torch.flip(x, dims=[0])
        return x

class RandomDropFrames(nn.Module):
    """Randomly drop frames and repeat last frame to maintain length."""
    def __init__(self, max_drop=2, p=0.5):
        super().__init__()
        self.max_drop = max_drop
        self.p = p
    def forward(self, x):
        if random.random() < self.p:
            T = x.shape[0]
            drop_count = random.randint(1, min(self.max_drop, T-1))
            drop_indices = sorted(random.sample(range(T), drop_count))
            keep_indices = list(range(T))
            for idx in drop_indices:
                keep_indices.remove(idx)
            while len(keep_indices) < T:
                keep_indices.append(keep_indices[-1])
            return x[keep_indices]
        return x

class Cutout(nn.Module):
    """Apply cutout augmentation to video frames."""
    def __init__(self, size=10, count=2, p=0.5):
        super().__init__()
        self.size = size
        self.count = count
        self.p = p
    def forward(self, x):
        if random.random() < self.p:
            T, C, H, W = x.shape
            mask = torch.ones_like(x)
            for _ in range(self.count):
                y = random.randint(0, H - self.size)
                x_pos = random.randint(0, W - self.size)
                mask[:, :, y:y+self.size, x_pos:x_pos+self.size] = 0
            return x * mask
        return x

# --- Focal Loss Implementation ---
# (Adapted from https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py)
class FocalLoss(nn.Module):
    """Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.
    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha=None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


# --- Helper Functions ---
def load_yaml(file_path):
    """Load a YAML configuration file."""
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# --- Dataset Definition ---
class FusionDataset(Dataset):
    # Removed hubert_embeddings_path, hubert_splits_dir
    # Added audio_base_path, hubert_model_name
    def __init__(self, manifest_file, audio_base_path, use_video, hubert_model_name="facebook/hubert-base-ls960", split='train', frames=48, img_size=112, config=None, augment=True):
        self.manifest_file = manifest_file
        self.audio_base_path = Path(audio_base_path) # Base path for audio files
        self.use_video = use_video # Store the flag
        self.split = split
        self.frames = frames
        self.img_size = img_size
        self.augment = augment and split == 'train'
        self.config = config or {}
        aug_config = self.config.get('augmentation', {}) # Reuse SlowFast aug config if needed

        # Load manifest
        print(f"Loading full manifest for split '{split}': {manifest_file}")
        df_full = pd.read_csv(manifest_file)

        # Select rows where the 'split' column *contains* the target split name (e.g., 'train', 'val')
        df = df_full[df_full['split'].str.contains(split, na=False)].reset_index(drop=True)
        print(f"Filtered manifest for split '{split}' using contains: {len(df)} rows")

        # Filter for known emotion categories
        df = df[df['label'].isin(EMOTION_LABELS)]
        self.label_to_idx = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
        df['manifest_label_idx'] = df['label'].map(self.label_to_idx) # Store manifest label index
        self.data = df # self.data now contains only rows for the current split

        # --- Initialize Hubert Feature Extractor ---
        print(f"Initializing Hubert Feature Extractor for {hubert_model_name}...")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(hubert_model_name)
        self.target_sr = self.feature_extractor.sampling_rate # Should be 16000
        # Hardcode max audio length for padding/truncation
        self.max_audio_len = 250000 # Approx 15.6s at 16kHz. Adjust if needed.
        print(f"Using hardcoded max_audio_len: {self.max_audio_len}")
        print(f"Target sampling rate: {self.target_sr}")
        # --- End Initialize Hubert Feature Extractor ---

        # Setup video transforms only if video is used
        if self.use_video:
            self.transform, self.temporal_augmentations = self._get_video_transforms(aug_config)
        else:
            self.transform, self.temporal_augmentations = None, None

    def _get_video_transforms(self, aug_config):
        """Create augmentation pipeline based on config (from train_slowfast_emotion.py)."""
        transform_list = []
        temporal_augmentations = [] # Augmentations applied to the final tensor

        # Basic transforms (applied per frame)
        transform_list.extend([
            transforms.ToTensor(), # Converts PIL Image to [C, H, W] tensor
        ])

        if self.augment:
            # Horizontal flip
            if aug_config.get('horizontal_flip', True):
                transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

            # Random affine transform
            affine_config = aug_config.get('random_affine', {})
            if affine_config.get('enabled', False):
                degrees = affine_config.get('rotate', 5)
                scale = (1-affine_config.get('scale', 0.05), 1+affine_config.get('scale', 0.05))
                translate_ratio = affine_config.get('translate', [0.05, 0.05])
                # Translate needs to be in pixels, not ratio for RandomAffine
                translate_pixels = (int(translate_ratio[0] * self.img_size), int(translate_ratio[1] * self.img_size))

                transform_list.append(
                    transforms.RandomAffine(
                        degrees=degrees,
                        translate=translate_pixels,
                        scale=scale,
                    )
                )

            # Color jitter
            jitter_config = aug_config.get('color_jitter', {})
            if jitter_config.get('enabled', False):
                transform_list.append(
                    transforms.ColorJitter(
                        brightness=jitter_config.get('brightness', 0.2),
                        contrast=jitter_config.get('contrast', 0.2),
                        saturation=jitter_config.get('saturation', 0.2),
                        hue=jitter_config.get('hue', 0.1)
                    )
                )

            # Temporal augmentations (applied after stacking frames)
            cutout_config = aug_config.get('cutout', {})
            if cutout_config.get('enabled', False):
                 temporal_augmentations.append(Cutout(
                     size=cutout_config.get('size', 10),
                     count=cutout_config.get('count', 2),
                     p=0.5
                 ))

            if aug_config.get('time_reverse', False):
                 temporal_augmentations.append(RandomTimeReverse(p=0.5))

            drop_config = aug_config.get('drop_frames', {})
            if drop_config.get('enabled', False):
                 temporal_augmentations.append(RandomDropFrames(
                     max_drop=drop_config.get('max_drop', 2),
                     p=0.5
                 ))

        # Normalization (always applied)
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )

        return transforms.Compose(transform_list), nn.Sequential(*temporal_augmentations)


    def _load_video_frames(self, video_path):
        """Load frames from a video file (adapted from train_slowfast_emotion.py)."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if total_frames <= 0 or fps <= 0:
            print(f"Warning: Invalid video {video_path}. Using dummy frames.")
            return torch.zeros(self.frames, 3, self.img_size, self.img_size)

        # Use center sampling for validation/test, random sampling for train
        if self.split == 'train':
            # Randomly sample start frame
            max_start_frame = max(0, total_frames - self.frames)
            start_frame = random.randint(0, max_start_frame)
        else:
            # Center crop
            start_frame = max(0, (total_frames - self.frames) // 2)

        end_frame = min(total_frames, start_frame + self.frames)

        # Ensure we sample exactly self.frames
        indices = np.linspace(start_frame, end_frame - 1, self.frames, dtype=np.int32)

        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                if frames: # Repeat last frame if read fails
                    frames.append(frames[-1].copy())
                else: # If first frame fails, use black frame
                    frames.append(np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8))
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.img_size, self.img_size))
            frames.append(frame)
        cap.release()

        # Apply per-frame transforms
        transformed_frames = [self.transform(Image.fromarray(f)) for f in frames]
        video_tensor = torch.stack(transformed_frames) # [T, C, H, W]

        # Apply temporal augmentations
        if self.augment and self.temporal_augmentations:
             video_tensor = self.temporal_augmentations(video_tensor)

        return video_tensor

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        audio_relative_path_str = item['path'] # Treat 'path' column as relative AUDIO path
        manifest_label_idx = item['manifest_label_idx']

        # Initialize audio tensors
        audio_input_values = None
        audio_attention_mask = None
        # default_len = 250000 # Example length for fallback - Now using self.max_audio_len

        # --- Load and Process Audio ---
        # Construct Full Audio Path (Handle dataset-specific structures)
        if "ravdess/AudioWAV" in audio_relative_path_str:
            # RAVDESS audio is in datasets/ravdess_videos/Actor_XX/
            corrected_audio_relative_str = audio_relative_path_str.replace("ravdess/AudioWAV", "ravdess_videos", 1)
            audio_full_path = self.audio_base_path / corrected_audio_relative_str
        elif "crema_d/AudioWAV" in audio_relative_path_str:
             # Crema-D audio is in datasets/crema_d_videos/
             corrected_audio_relative_str = audio_relative_path_str.replace("crema_d/AudioWAV/", "crema_d_videos/", 1)
             audio_full_path = self.audio_base_path / corrected_audio_relative_str
        elif "crema_d/" in audio_relative_path_str: # Fallback if AudioWAV wasn't in path
             corrected_audio_relative_str = audio_relative_path_str.replace("crema_d/", "crema_d_videos/", 1)
             audio_full_path = self.audio_base_path / corrected_audio_relative_str
        else:
             # Default assumption if not RAVDESS or Crema-D
             audio_relative_path = Path(audio_relative_path_str) # Keep original relative path for default
             audio_full_path = self.audio_base_path / audio_relative_path

        try:
            waveform, sample_rate = torchaudio.load(audio_full_path)
            # Resample if necessary
            if sample_rate != self.target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sr)
                waveform = resampler(waveform)

            # Process with feature extractor (handles padding/truncation)
            # Squeeze waveform if it has a channel dimension (torchaudio loads [C, T])
            processed_audio = self.feature_extractor(
                waveform.squeeze(0),
                sampling_rate=self.target_sr,
                return_tensors="pt",
                padding='max_length',       # Pad to max_length
                truncation=True,            # Truncate if longer
                max_length=self.max_audio_len, # Use the determined max length
                return_attention_mask=True # Explicitly request attention mask
            )
            # Assign results directly to the pre-initialized variables
            audio_input_values = processed_audio.input_values.squeeze(0)
            # Check if attention_mask was returned before accessing it
            if 'attention_mask' in processed_audio:
                audio_attention_mask = processed_audio.attention_mask.squeeze(0)
            else:
                # Create a default mask if not returned (e.g., all ones)
                print(f"Warning: 'attention_mask' not found in processed_audio for {audio_full_path}. Creating default mask.")
                audio_attention_mask = torch.ones_like(audio_input_values, dtype=torch.long)


        except Exception as e:
            print(f"Error loading or processing audio {audio_full_path}. Returning zeros.") # Simplified message
            print("--- Traceback ---")
            traceback.print_exc() # Print the full traceback
            print("--- End Traceback ---")
            # Determine expected input length (might need adjustment based on feature extractor)
            # For Hubert base, a common max length is ~250000 samples (around 15.6s)
            # Let's use a reasonable default, but this might need tuning.
            # Alternatively, get max length from feature_extractor if available.
            # default_len = 250000 # Example length - Defined earlier - Now using self.max_audio_len
            # Assign dummy tensors directly to the pre-initialized variables, using the consistent max length
            audio_input_values = torch.zeros(self.max_audio_len)
            audio_attention_mask = torch.zeros(self.max_audio_len, dtype=torch.long)
            # Note: We don't need hubert_label_idx here, using manifest_label_idx directly

        # --- Load Video ---
        if self.use_video:
            try:
                # Derive video path using the same logic as verify_paths_exist.py
                # Note: We assume video_base_path is the same as audio_base_path for simplicity here,
                # as passed in the command line args. If they differ, this needs adjustment.
                video_base_path = self.audio_base_path # Assuming same base for video derivation

                derived_video_full_path = None # Initialize
                vid_full_mp4 = None # Initialize to avoid reference before assignment in except block

                if "ravdess/AudioWAV" in audio_relative_path_str:
                    # RAVDESS Video: Look for .mp4 in datasets/ravdess_videos/
                    vid_rel_mp4_str = audio_relative_path_str.replace("ravdess/AudioWAV", "ravdess_videos", 1).replace(".wav", ".mp4")
                    vid_full_mp4 = video_base_path / vid_rel_mp4_str
                    if vid_full_mp4.exists():
                        derived_video_full_path = vid_full_mp4
                    else:
                        # Try AVI as fallback for RAVDESS
                        vid_rel_avi_str = audio_relative_path_str.replace("ravdess/AudioWAV", "ravdess_videos", 1).replace(".wav", ".avi")
                        vid_full_avi = video_base_path / vid_rel_avi_str
                        if vid_full_avi.exists():
                             derived_video_full_path = vid_full_avi
                        else:
                             raise FileNotFoundError(f"RAVDESS video not found for {audio_relative_path_str}. Searched MP4/AVI at {vid_full_mp4} and {vid_full_avi}")

                elif "crema_d/AudioWAV" in audio_relative_path_str:
                     # Crema-D Video: Look for .flv in datasets/crema_d_videos/
                     vid_rel_flv_str = audio_relative_path_str.replace("crema_d/AudioWAV/", "crema_d_videos/", 1).replace(".wav", ".flv")
                     vid_full_flv = video_base_path / vid_rel_flv_str
                     if vid_full_flv.exists():
                         derived_video_full_path = vid_full_flv
                     else:
                         # Raise error if FLV not found
                         raise FileNotFoundError(f"Crema-D video (.flv) not found for {audio_relative_path_str} at {vid_full_flv}")
                elif "crema_d/" in audio_relative_path_str: # Fallback if AudioWAV wasn't in path
                     vid_rel_flv_str = audio_relative_path_str.replace("crema_d/", "crema_d_videos/", 1).replace(".wav", ".flv")
                     vid_full_flv = video_base_path / vid_rel_flv_str
                     if vid_full_flv.exists():
                         derived_video_full_path = vid_full_flv
                     else:
                         raise FileNotFoundError(f"Crema-D video (.flv) not found for {audio_relative_path_str} at {vid_full_flv}")

                else:
                    # Fallback/Default derivation (if needed for other datasets)
                    # This part might need adjustment based on actual structure of other datasets
                    audio_relative_path = Path(audio_relative_path_str) # Need Path object for stem
                    vid_rel_mp4_str = audio_relative_path.with_suffix('.mp4').name # Simple suffix change
                    # Construct path relative to some assumed video dir structure? Needs clarification.
                    # Example: Assuming video is in a parallel 'VideoMP4' dir relative to audio base
                    # This is highly speculative without knowing the structure
                    # video_full_path_default = video_base_path / audio_relative_path.parent.parent / "VideoMP4" / vid_rel_mp4_str
                    # For now, let's assume a simpler structure or raise error
                    raise NotImplementedError(f"Video path derivation logic not defined for non-RAVDESS/Crema-D path: {audio_relative_path_str}")


                # Load the video frames using the determined path
                if derived_video_full_path and derived_video_full_path.exists():
                    # Stem check (optional but good practice)
                    audio_stem = Path(audio_relative_path_str).stem
                    video_stem = derived_video_full_path.stem
                    if audio_stem != video_stem:
                         warnings.warn(f"Stem mismatch! Audio: {audio_stem}, Video: {video_stem} for {derived_video_full_path}")

                    video_tensor = self._load_video_frames(str(derived_video_full_path))
                else:
                    # This should ideally not be reached if FileNotFoundError is raised above correctly
                    raise FileNotFoundError(f"Derived video path {derived_video_full_path} does not exist.")

            except Exception as e_vid:
                 print(f"Error deriving or loading video for audio {audio_full_path}: {e_vid}. Using dummy video.")
                 video_tensor = torch.zeros(self.frames, 3, self.img_size, self.img_size) # Fallback

        else:
            video_tensor = None # Video not used


        # Return video_tensor, audio tensors, and label
        return video_tensor, audio_input_values, audio_attention_mask, manifest_label_idx # Always use manifest label


# --- Model Definition ---

class FusionModel(nn.Module):
    def __init__(self, num_classes, video_checkpoint_path, hubert_checkpoint_path, hubert_model_name="facebook/hubert-base-ls960", fusion_dim=512, dropout=0.5):
        super().__init__()
        self.num_classes = num_classes

        # --- Video Branch ---
        print("Instantiating VideoEmotionClassifier (R3D-18 based)...")
        self.video_embedder = VideoEmotionClassifier(
            num_classes=len(EMOTION_LABELS), # Match the original training
            use_se=True, # Assuming SE blocks were used
            pretrained=False
        )
        self.video_embedding_size = self.video_embedder.embedding_size # Should be 512

        print(f"Loading Video (R3D-18) weights from checkpoint: {video_checkpoint_path}")
        if video_checkpoint_path and os.path.exists(video_checkpoint_path):
            video_checkpoint = torch.load(video_checkpoint_path, map_location='cpu')
            video_state_dict = video_checkpoint.get('model_state_dict', video_checkpoint)
            load_result_vid = self.video_embedder.load_state_dict(video_state_dict, strict=False)
            print(f"Loaded Video Embedder state_dict. Load result: {load_result_vid}")
        else:
            print("Warning: Video checkpoint not provided or file not found. "
                  "Proceeding with randomly initialized video backbone.")
        self.video_embedder.classifier = nn.Identity() # Remove final layer
        print("Freezing Video (R3D-18) backbone parameters.")
        for param in self.video_embedder.video_embedder.parameters():
            param.requires_grad = False

        # --- Audio Branch (Hubert SER) ---
        print(f"Instantiating HubertSER model ({hubert_model_name})...")
        # Instantiate HubertSER - num_classes might differ, handle during loading
        self.hubert_ser = HubertSER(
            hubert_name=hubert_model_name,
            num_classes=num_classes # Use target num_classes initially
        )
        print(f"Loading Hubert SER weights from checkpoint: {hubert_checkpoint_path}")
        if hubert_checkpoint_path and os.path.exists(hubert_checkpoint_path):
            hub_checkpoint = torch.load(hubert_checkpoint_path, map_location='cpu')
        else:
            print("Warning: Hubert checkpoint not provided or file not found. "
                  "Proceeding with randomly initialized HuBERT branch.")
            hub_checkpoint = None  # Skip loading
        if hub_checkpoint is not None:
            # Check if it's a PL checkpoint with 'state_dict' key
            if 'state_dict' in hub_checkpoint:
                hub_state_dict = hub_checkpoint['state_dict']
            else:
                hub_state_dict = hub_checkpoint # Assume it's just the state dict

            # Adjust keys if needed (e.g., remove 'model.' prefix if saved directly)
            # hub_state_dict = {k.replace("model.", ""): v for k, v in hub_state_dict.items()}

            # Load weights, potentially ignoring the final classifier if sizes mismatch
            load_result_hub = self.hubert_ser.load_state_dict(hub_state_dict, strict=False)
            print(f"Loaded Hubert SER state_dict. Load result (strict=False): {load_result_hub}")
            if load_result_hub.missing_keys or load_result_hub.unexpected_keys:
                print("  Note: Mismatched keys likely due to loading a pre-trained model. Check carefully.")
                print(f"  Missing keys: {load_result_hub.missing_keys}")
                print(f"  Unexpected keys: {load_result_hub.unexpected_keys}")

        # Get feature dimension *before* the final FC layer of HubertSER
        self.hubert_feature_dim = self.hubert_ser.fc.in_features # Get dim before replacing
        self.hubert_feature_dim = self.hubert_ser.fc.in_features # Get dim before replacing
        self.hubert_ser.fc = nn.Identity() # Remove final layer

        # Freeze Hubert backbone by default, allow unfreezing last N layers
        print("Freezing Hubert backbone parameters by default.")
        for param in self.hubert_ser.hubert.parameters():
            param.requires_grad = False

        # Unfreeze last N layers if requested (handle potential errors)
        hubert_unfreeze_layers = getattr(self, '_hubert_unfreeze_layers', 0) # Get from instance attr if set
        if hubert_unfreeze_layers > 0:
            print(f"Unfreezing the last {hubert_unfreeze_layers} Hubert transformer layers and the final projection layer.")
            try:
                # Ensure layers exist before trying to unfreeze
                if hasattr(self.hubert_ser.hubert.encoder, 'layers'):
                    num_hubert_layers = len(self.hubert_ser.hubert.encoder.layers)
                    if hubert_unfreeze_layers > num_hubert_layers:
                        print(f"Warning: Requested to unfreeze {hubert_unfreeze_layers} layers, but Hubert model only has {num_hubert_layers}. Unfreezing all layers.")
                        hubert_unfreeze_layers = num_hubert_layers

                    for i in range(num_hubert_layers - hubert_unfreeze_layers, num_hubert_layers):
                        print(f"  Unfreezing Hubert layer {i}")
                        for param in self.hubert_ser.hubert.encoder.layers[i].parameters():
                            param.requires_grad = True

                    # Also unfreeze the final layer norm and projection if they exist
                    if hasattr(self.hubert_ser.hubert.encoder, 'layer_norm'):
                         print("  Unfreezing Hubert final encoder layer norm")
                         for param in self.hubert_ser.hubert.encoder.layer_norm.parameters():
                             param.requires_grad = True
                    if hasattr(self.hubert_ser.hubert, 'project_hid'): # Check if projection layer exists
                         print("  Unfreezing Hubert final projection layer")
                         for param in self.hubert_ser.hubert.project_hid.parameters():
                             param.requires_grad = True
                    # Unfreeze the feature projection layer as well if it exists
                    if hasattr(self.hubert_ser.hubert, 'feature_projection'):
                        print("  Unfreezing Hubert feature projection layer")
                        for param in self.hubert_ser.hubert.feature_projection.parameters():
                            param.requires_grad = True

                else:
                    print("Warning: Could not find 'layers' attribute in hubert.encoder. Cannot unfreeze specific layers.")

            except AttributeError as e:
                print(f"Warning: Error accessing Hubert layers for unfreezing: {e}. Keeping all backbone layers frozen.")

        # --- Fusion Layer ---
        # Use the determined feature dimensions
        print(f"Video embedding size: {self.video_embedding_size}, Hubert feature dim: {self.hubert_feature_dim}")
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.video_embedding_size + self.hubert_feature_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Final Classifier
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(self, video_input, audio_input_values, audio_attention_mask):
        # video_input shape: [B, T, C, H, W]
        # audio_input_values shape: [B, AudioSeqLen]
        # audio_attention_mask shape: [B, AudioSeqLen]

        # Get Video features
        video_features = self.video_embedder(video_input) # Shape: [B, video_embedding_size]

        # Get Hubert features (before final FC)
        # Pass audio data through the Hubert backbone
        hubert_outputs = self.hubert_ser.hubert(input_values=audio_input_values, attention_mask=audio_attention_mask)
        # Pool the hidden states using the method defined in HubertSER
        hubert_features = self.hubert_ser._pool(hubert_outputs.last_hidden_state, audio_attention_mask) # Shape: [B, hubert_feature_dim]
        # Note: We are NOT applying the dropout from HubertSER here, as dropout is in the fusion layer.

        # Concatenate features
        fused_features = torch.cat((video_features, hubert_features), dim=1)

        # Pass through fusion layer
        fused_output = self.fusion_layer(fused_features)

        # Classify
        logits = self.classifier(fused_output)

        return logits

# --- Training and Validation Functions (Placeholders) ---
def train_epoch(model, dataloader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    # Ensure parts of Hubert backbone remain in eval mode if *partially* frozen
    if hasattr(model, 'hubert_ser'):
        # Set the main backbone to eval if *any* part is frozen
        # This prevents BatchNorm/Dropout updates in the frozen parts.
        # The un-frozen layers will still compute gradients.
        if not all(p.requires_grad for p in model.hubert_ser.hubert.parameters()):
             model.hubert_ser.hubert.eval()
             print("Note: Hubert backbone set to eval() mode because some layers are frozen.")
        else:
             model.hubert_ser.hubert.train() # Ensure it's training if fully unfrozen


    running_loss = 0.0
    all_preds = []
    all_labels = []
    progress_bar = tqdm(dataloader, desc="Training Fusion")

    # Adjusted loop to unpack audio tensors
    for video_data, audio_input_values, audio_attention_mask, labels in progress_bar:
        video_data = video_data.to(device) if video_data is not None else None
        audio_input_values = audio_input_values.to(device)
        audio_attention_mask = audio_attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Use torch.amp.autocast directly
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            # Pass audio tensors to the model
            outputs = model(video_data, audio_input_values, audio_attention_mask)
            loss = criterion(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        # Update progress bar display if needed

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds) * 100
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    # Use leave=False for cleaner logs after completion
    progress_bar = tqdm(dataloader, desc="Validating Fusion", leave=False)

    with torch.no_grad():
        # Adjusted loop to unpack audio tensors
        for video_data, audio_input_values, audio_attention_mask, labels in progress_bar:
            video_data = video_data.to(device) if video_data is not None else None
            audio_input_values = audio_input_values.to(device)
            audio_attention_mask = audio_attention_mask.to(device)
            labels = labels.to(device)

            # Pass audio tensors to the model
            outputs = model(video_data, audio_input_values, audio_attention_mask)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / len(dataloader)
    val_acc = accuracy_score(all_labels, all_preds) * 100
    print("\nValidation Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=EMOTION_LABELS, zero_division=0))
    return val_loss, val_acc

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Train Multimodal Fusion Model")
    # Paths
    parser.add_argument("--manifest_file", type=str, required=True, help="Path to main CSV manifest (must contain video paths, labels, splits)")
    # Removed hubert_embeddings_path and hubert_splits_dir
    parser.add_argument("--audio_base_path", type=str, required=True, help="Base directory where audio files (relative paths in manifest) are located")
    parser.add_argument("--hubert_checkpoint", type=str, required=True, help="Path to the pre-trained Hubert SER model checkpoint (.ckpt)") # Added
    parser.add_argument("--video_checkpoint", type=str, default=None, help="Path to the pre-trained Video (R3D-18) model checkpoint (.pt). Required if --use_video is set.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save fusion models and logs")
    parser.add_argument("--config", type=str, default=None, help="Optional path to YAML config file for detailed settings")
    # Set default=True for use_video as the model requires it
    parser.add_argument("--use_video", action=argparse.BooleanOptionalAction, default=True, help="Include video features in the fusion model (default: True)")
    # parser.add_argument("--use_hubert", action="store_true", default=True, help="Include hubert features (default: True, required for now)") # Removed, implicitly true now

    # Model Hyperparameters
    parser.add_argument("--hubert_model_name", type=str, default="facebook/hubert-base-ls960", help="Name of the Hubert model used for the checkpoint (for feature extractor)") # Added
    parser.add_argument("--hubert_unfreeze_layers", type=int, default=0, help="Number of final Hubert transformer layers to unfreeze (default: 0)") # Added
    parser.add_argument("--fusion_dim", type=int, default=512, help="Dimension of the fusion layer output")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate for fusion/classifier layers")

    # Training Hyperparameters
    parser.add_argument("--criterion", type=str, default="cross_entropy", choices=["cross_entropy", "focal"], help="Loss function (default: cross_entropy)") # Added
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Gamma parameter for Focal Loss (default: 2.0)") # Added
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine", "plateau"], help="Learning rate scheduler (default: none)") # Added
    parser.add_argument("--cosine_t0", type=int, default=10, help="T_0 for CosineAnnealingWarmRestarts (default: 10)") # Added
    parser.add_argument("--plateau_patience", type=int, default=5, help="Patience for ReduceLROnPlateau (default: 5)") # Added
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing factor")
    parser.add_argument("--early_stop", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")

    # Data/Loader Settings
    parser.add_argument("--img_size", type=int, default=112, help="Image size for video frames")
    parser.add_argument("--frames", type=int, default=48, help="Number of frames per video clip")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--lr_backbone", type=float, default=None, help="Optional different learning rate for unfrozen backbone layers") # Added

    args = parser.parse_args()

    # --- Argument Validation ---
    if args.use_video and not args.video_checkpoint:
        # This check might be redundant now with default=True, but keep for safety
        parser.error("--video_checkpoint is required when --use_video is set.")
    if not args.hubert_checkpoint: # Ensure Hubert checkpoint is provided
        parser.error("--hubert_checkpoint is required.")
    if not args.use_video:
        print("Warning: --no-use_video was specified, but the current model architecture requires video input. Training will likely fail.")
    # --- End Argument Validation ---

    # Load config if provided, otherwise use args
    config = {}
    if args.config and os.path.exists(args.config):
        print(f"Loading config from {args.config}")
        config = load_yaml(args.config)
    # TODO: Override config with args if necessary

    # Setup device, output dir, etc.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Using device: {device}")

    # Create Datasets and Dataloaders
    print("Creating Datasets...")
    train_dataset = FusionDataset(
        manifest_file=args.manifest_file,
        audio_base_path=args.audio_base_path, # Use new arg
        hubert_model_name=args.hubert_model_name, # Pass model name for extractor
        split="train",
        frames=args.frames,
        img_size=args.img_size,
        use_video=args.use_video,
        config=config,
        augment=True
    )
    val_dataset = FusionDataset(
        manifest_file=args.manifest_file,
        audio_base_path=args.audio_base_path, # Use new arg
        hubert_model_name=args.hubert_model_name, # Pass model name for extractor
        split="val",
        frames=args.frames,
        img_size=args.img_size,
        use_video=args.use_video,
        config=config,
        augment=False
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print("Datasets created.")

    # Calculate class weights if using Focal Loss
    class_weights = None
    if args.criterion == "focal":
        print("Calculating class weights for Focal Loss...")
        all_labels = train_dataset.data['manifest_label_idx'].tolist()
        label_counts = Counter(all_labels)
        total_samples = len(all_labels)
        weights = []
        # Ensure weights are calculated in the order of EMOTION_LABELS
        for i in range(len(EMOTION_LABELS)):
            count = label_counts.get(i, 0) # Get count for class index i
            weight = total_samples / (len(EMOTION_LABELS) * count) if count > 0 else 1.0 # Inverse frequency
            weights.append(weight)
        class_weights = torch.tensor(weights, dtype=torch.float).to(device)
        print(f"Class weights: {class_weights}")


    # Create Model, passing the actual dimension
    print("Creating Fusion Model...")
    # Pass unfreeze layers arg to model constructor indirectly via an attribute
    # This is a bit hacky but avoids changing the constructor signature drastically
    FusionModel._hubert_unfreeze_layers = args.hubert_unfreeze_layers
    model = FusionModel(
        num_classes=len(EMOTION_LABELS),
        video_checkpoint_path=args.video_checkpoint,
        hubert_checkpoint_path=args.hubert_checkpoint, # Pass Hubert checkpoint path
        hubert_model_name=args.hubert_model_name, # Pass model name
        fusion_dim=args.fusion_dim,
        dropout=args.dropout
    ).to(device)
    del FusionModel._hubert_unfreeze_layers # Clean up the temporary attribute
    print("Model created.")

    # Optimizer (potentially with differential LR), Scheduler, Loss
    print("Setting up optimizer, scheduler, and loss...")
    optimizer_params = []
    if args.hubert_unfreeze_layers > 0 and args.lr_backbone is not None:
        print(f"Using differential learning rate: Backbone LR={args.lr_backbone}, Head LR={args.lr}")
        # Separate backbone (unfrozen Hubert) params from head params
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if name.startswith("hubert_ser.hubert"):
                    backbone_params.append(param)
                else:
                    head_params.append(param)
        optimizer_params = [
            {'params': backbone_params, 'lr': args.lr_backbone},
            {'params': head_params, 'lr': args.lr}
        ]
    else:
        print(f"Using single learning rate: {args.lr}")
        optimizer_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler
    if args.scheduler == "cosine":
        print(f"Using CosineAnnealingWarmRestarts scheduler with T_0={args.cosine_t0}")
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.cosine_t0, eta_min=args.lr * 0.01) # Example eta_min
    elif args.scheduler == "plateau":
        print(f"Using ReduceLROnPlateau scheduler with patience={args.plateau_patience}")
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.plateau_patience, verbose=True) # Monitor val_acc
    else:
        print("No learning rate scheduler selected.")
        scheduler = None

    # Loss Function
    if args.criterion == "focal":
        print(f"Using Focal Loss with gamma={args.focal_gamma} and calculated alpha weights.")
        criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
    else:
        print(f"Using Cross Entropy Loss with label smoothing={args.label_smoothing}")
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None # Use full path

    # Training Loop
    best_val_acc = 0.0
    patience_counter = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"fusion_model_{timestamp}"

    print("Starting training loop...")
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device, args.fp16)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}% | Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            print(f"Saving best model with Val Acc: {best_val_acc:.2f}%")
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"{model_name}_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop:
                print(f"Early stopping triggered after {args.early_stop} epochs without improvement.")
                break

        # Step the scheduler
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_acc) # Plateau needs the metric
            else:
                scheduler.step() # Others step per epoch

    print(f"\nTraining finished. Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved to: {os.path.join(args.output_dir, f'{model_name}_best.pt')}")

if __name__ == "__main__":
    main()
