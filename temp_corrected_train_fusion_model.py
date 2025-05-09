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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, BatchSampler # Added BatchSampler
from torch.cuda.amp import GradScaler # Removed autocast import here, will use torch.amp.autocast
import torchvision
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau # Added CosineAnnealingWarmRestarts, ReduceLROnPlateau
import torchaudio # For audio loading
from collections import Counter # For class weights
print(f"Using torchaudio backend: {torchaudio.get_audio_backend()}") # Print the actual backend
from transformers import AutoFeatureExtractor # For audio processing

# Try importing sklearn metrics, warn if not found
try:
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, average_precision_score, precision_recall_fscore_support
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not found. Metrics like AUROC, PR-AUC, and detailed classification reports will not be available.")
    SKLEARN_AVAILABLE = False
    # Define dummy functions if sklearn is not available
    def accuracy_score(y_true, y_pred): return np.mean(np.array(y_true) == np.array(y_pred))
    def classification_report(y_true, y_pred, target_names=None, zero_division=0): return "Classification report requires scikit-learn."
    def roc_auc_score(y_true, y_score): return 0.0
    def average_precision_score(y_true, y_score): return 0.0
    def precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0): return (0.0, 0.0, 0.0, None)


from datetime import datetime

# Import the correct classifier from the script that generated the checkpoint
try:
    # Use absolute import assuming script is run from conjunction-train root
    from scripts.train_slowfast_emotion import EmotionClassifier as VideoEmotionClassifier
    print("Successfully imported EmotionClassifier from scripts.train_slowfast_emotion")
except ImportError as e:
    print(f"Error importing EmotionClassifier from scripts.train_slowfast_emotion: {e}")
    print("Ensure train_slowfast_emotion.py is in the 'scripts' directory relative to the execution path.")
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

# Import Dataloaders and Samplers
try:
    from dataloaders.humor_dataset import HumorDataset
    from samplers.balanced_humor_sampler import BalancedHumorSampler
    print("Successfully imported HumorDataset and BalancedHumorSampler.")
    HUMOR_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import humor dataset/sampler: {e}. --add_humor flag will fail if used.")
    HumorDataset = None # Define as None to allow script to load without error if not used
    BalancedHumorSampler = None
    HUMOR_COMPONENTS_AVAILABLE = False

# Loss for humor head
from torch.nn import BCEWithLogitsLoss


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

# --- Dataset Definition (Original FusionDataset - used if --add_humor is False) ---
class FusionDataset(Dataset):
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

        # --- Load and Process Audio ---
        # Construct Full Audio Path (Handle dataset-specific structures)
        if "ravdess/AudioWAV" in audio_relative_path_str:
            corrected_audio_relative_str = audio_relative_path_str.replace("ravdess/AudioWAV", "ravdess_videos", 1)
            audio_full_path = self.audio_base_path / corrected_audio_relative_str
        elif "crema_d/AudioWAV" in audio_relative_path_str:
             corrected_audio_relative_str = audio_relative_path_str.replace("crema_d/AudioWAV/", "crema_d_videos/", 1)
             audio_full_path = self.audio_base_path / corrected_audio_relative_str
        elif "crema_d/" in audio_relative_path_str: # Fallback if AudioWAV wasn't in path
             corrected_audio_relative_str = audio_relative_path_str.replace("crema_d/", "crema_d_videos/", 1)
             audio_full_path = self.audio_base_path / corrected_audio_relative_str
        else:
             audio_relative_path = Path(audio_relative_path_str)
             audio_full_path = self.audio_base_path / audio_relative_path

        try:
            waveform, sample_rate = torchaudio.load(audio_full_path)
            if sample_rate != self.target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sr)
                waveform = resampler(waveform)

            processed_audio = self.feature_extractor(
                waveform.squeeze(0),
                sampling_rate=self.target_sr,
                return_tensors="pt",
                padding='max_length',
                truncation=True,
                max_length=self.max_audio_len,
                return_attention_mask=True
            )
            audio_input_values = processed_audio.input_values.squeeze(0)
            if 'attention_mask' in processed_audio:
                audio_attention_mask = processed_audio.attention_mask.squeeze(0)
            else:
                print(f"Warning: 'attention_mask' not found in processed_audio for {audio_full_path}. Creating default mask.")
                audio_attention_mask = torch.ones_like(audio_input_values, dtype=torch.long)

        except Exception as e:
            print(f"Error loading or processing audio {audio_full_path}. Returning zeros.")
            traceback.print_exc()
            audio_input_values = torch.zeros(self.max_audio_len)
            audio_attention_mask = torch.zeros(self.max_audio_len, dtype=torch.long)

        # --- Load Video ---
        video_tensor = None # Initialize
        if self.use_video:
            try:
                video_base_path = self.audio_base_path # Assuming same base for video derivation
                derived_video_full_path = None

                if "ravdess/AudioWAV" in audio_relative_path_str:
                    vid_rel_mp4_str = audio_relative_path_str.replace("ravdess/AudioWAV", "ravdess_videos", 1).replace(".wav", ".mp4")
                    vid_full_mp4 = video_base_path / vid_rel_mp4_str
                    if vid_full_mp4.exists():
                        derived_video_full_path = vid_full_mp4
                    else:
                        vid_rel_avi_str = audio_relative_path_str.replace("ravdess/AudioWAV", "ravdess_videos", 1).replace(".wav", ".avi")
                        vid_full_avi = video_base_path / vid_rel_avi_str
                        if vid_full_avi.exists():
                             derived_video_full_path = vid_full_avi
                        else:
                             raise FileNotFoundError(f"RAVDESS video not found for {audio_relative_path_str}. Searched MP4/AVI at {vid_full_mp4} and {vid_full_avi}")

                elif "crema_d/AudioWAV" in audio_relative_path_str or "crema_d/" in audio_relative_path_str:
                     vid_rel_flv_str = audio_relative_path_str.replace("crema_d/AudioWAV/", "crema_d_videos/", 1).replace("crema_d/", "crema_d_videos/", 1).replace(".wav", ".flv")
                     vid_full_flv = video_base_path / vid_rel_flv_str
                     if vid_full_flv.exists():
                         derived_video_full_path = vid_full_flv
                     else:
                         raise FileNotFoundError(f"Crema-D video (.flv) not found for {audio_relative_path_str} at {vid_full_flv}")
                else:
                    raise NotImplementedError(f"Video path derivation logic not defined for non-RAVDESS/Crema-D path: {audio_relative_path_str}")

                if derived_video_full_path and derived_video_full_path.exists():
                    audio_stem = Path(audio_relative_path_str).stem
                    video_stem = derived_video_full_path.stem
                    if audio_stem != video_stem:
                         warnings.warn(f"Stem mismatch! Audio: {audio_stem}, Video: {video_stem} for {derived_video_full_path}")
                    video_tensor = self._load_video_frames(str(derived_video_full_path))
                else:
                    raise FileNotFoundError(f"Derived video path {derived_video_full_path} does not exist.")

            except Exception as e_vid:
                 print(f"Error deriving or loading video for audio {audio_full_path}: {e_vid}. Using dummy video.")
                 video_tensor = torch.zeros(self.frames, 3, self.img_size, self.img_size) # Fallback

        # Return video_tensor, audio tensors, and label
        return video_tensor, audio_input_values, audio_attention_mask, manifest_label_idx # Always use manifest label

    def get_class_weights(self):
        """Calculates class weights based on the 'manifest_label_idx' column."""
        if 'manifest_label_idx' not in self.data.columns:
            print("Warning: 'manifest_label_idx' column not found in dataset. Cannot calculate class weights.")
            return None
        label_counts = Counter(self.data['manifest_label_idx'])
        total_samples = len(self.data)
        num_classes = len(EMOTION_LABELS)
        weights = []
        for i in range(num_classes):
            count = label_counts.get(i, 0)
            weight = total_samples / (num_classes * count) if count > 0 else 1.0 # Inverse frequency
            weights.append(weight)
        return weights

    def collate_fn(self, batch):
        """Custom collate function to handle variable data types."""
        video_tensors, audio_input_values, audio_attention_masks, labels = zip(*batch)

        # Stack video tensors if they exist
        video_batch = None
        if video_tensors[0] is not None:
            video_batch = torch.stack(video_tensors)

        # Stack audio tensors
        audio_values_batch = torch.stack(audio_input_values)
        audio_mask_batch = torch.stack(audio_attention_masks)

        # Stack labels
        labels_batch = torch.tensor(labels, dtype=torch.long)

        return video_batch, audio_values_batch, audio_mask_batch, labels_batch


# --- Model Definition ---
class FusionModel(nn.Module):
    def __init__(self, num_classes, video_checkpoint_path, hubert_checkpoint_path, hubert_model_name="facebook/hubert-base-ls960", fusion_dim=512, dropout=0.5, hubert_unfreeze_layers=0):
        super().__init__()
        self.num_classes = num_classes
        self._hubert_unfreeze_layers = hubert_unfreeze_layers # Store unfreeze count

        # --- Video Branch ---
        print("Instantiating VideoEmotionClassifier (R3D-18 based)...")
        self.video_embedder = VideoEmotionClassifier(
            num_classes=len(EMOTION_LABELS), # Match the original training
            use_se=True, # Assuming SE blocks were used
            pretrained=False
        )
        self.video_embedding_size = self.video_embedder.embedding_size # Should be 512

        print(f"Loading Video (R3D-18) weights from checkpoint: {video_checkpoint_path}")
        if not os.path.exists(video_checkpoint_path):
            print(f"Error: Video Checkpoint file not found at {video_checkpoint_path}")
            sys.exit(1)
        video_checkpoint = torch.load(video_checkpoint_path, map_location='cpu')
        video_state_dict = video_checkpoint.get('model_state_dict', video_checkpoint)
        load_result_vid = self.video_embedder.load_state_dict(video_state_dict, strict=True)
        print(f"Loaded Video Embedder state_dict. Load result: {load_result_vid}")
        self.video_embedder.classifier = nn.Identity() # Remove final layer
        print("Freezing Video (R3D-18) backbone parameters.")
        for param in self.video_embedder.video_embedder.parameters():
            param.requires_grad = False

        # --- Audio Branch (Hubert SER) ---
        print(f"Instantiating HubertSER model ({hubert_model_name})...")
        self.hubert_ser = HubertSER(
            hubert_name=hubert_model_name,
            num_classes=num_classes # Use target num_classes initially
        )
        print(f"Loading Hubert SER weights from checkpoint: {hubert_checkpoint_path}")
        if not os.path.exists(hubert_checkpoint_path):
            print(f"Error: Hubert Checkpoint file not found at {hubert_checkpoint_path}")
            sys.exit(1)

        hub_checkpoint = torch.load(hubert_checkpoint_path, map_location='cpu')
        if 'state_dict' in hub_checkpoint:
            hub_state_dict = hub_checkpoint['state_dict']
        else:
            hub_state_dict = hub_checkpoint

        load_result_hub = self.hubert_ser.load_state_dict(hub_state_dict, strict=False)
        print(f"Loaded Hubert SER state_dict. Load result (strict=False): {load_result_hub}")
        if load_result_hub.missing_keys or load_result_hub.unexpected_keys:
            print("  Note: Mismatched keys likely due to loading a pre-trained model. Check carefully.")
            print(f"  Missing keys: {load_result_hub.missing_keys}")
            print(f"  Unexpected keys: {load_result_hub.unexpected_keys}")

        self.hubert_feature_dim = self.hubert_ser.fc.in_features
        self.hubert_ser.fc = nn.Identity() # Remove final layer

        # Freeze/Unfreeze Hubert layers
        self._configure_hubert_grads()

        # --- Fusion Layer ---
        print(f"Video embedding size: {self.video_embedding_size}, Hubert feature dim: {self.hubert_feature_dim}")
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.video_embedding_size + self.hubert_feature_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Emotion Classifier Head
        self.emotion_head = nn.Linear(fusion_dim, num_classes)
        # Humor Classifier Head (Binary)
        self.humor_head = nn.Linear(fusion_dim, 1) # Output a single logit for BCEWithLogitsLoss

    def _configure_hubert_grads(self):
        """Freezes or unfreezes Hubert layers based on self._hubert_unfreeze_layers."""
        print("Configuring Hubert parameter gradients...")
        # Freeze all by default
        for param in self.hubert_ser.hubert.parameters():
            param.requires_grad = False

        if self._hubert_unfreeze_layers > 0:
            print(f"Unfreezing the last {self._hubert_unfreeze_layers} Hubert transformer layers and the final projection layer.")
            try:
                if hasattr(self.hubert_ser.hubert.encoder, 'layers'):
                    num_hubert_layers = len(self.hubert_ser.hubert.encoder.layers)
                    if self._hubert_unfreeze_layers > num_hubert_layers:
                        print(f"Warning: Requested to unfreeze {self._hubert_unfreeze_layers} layers, but Hubert model only has {num_hubert_layers}. Unfreezing all layers.")
                        self._hubert_unfreeze_layers = num_hubert_layers

                    for i in range(num_hubert_layers - self._hubert_unfreeze_layers, num_hubert_layers):
                        print(f"  Unfreezing Hubert layer {i}")
                        for param in self.hubert_ser.hubert.encoder.layers[i].parameters():
                            param.requires_grad = True

                    # Also unfreeze the final layer norm and projection if they exist
                    if hasattr(self.hubert_ser.hubert.encoder, 'layer_norm'):
                         print("  Unfreezing Hubert final encoder layer norm")
                         for param in self.hubert_ser.hubert.encoder.layer_norm.parameters():
                             param.requires_grad = True
                    if hasattr(self.hubert_ser.hubert, 'project_hid'):
                         print("  Unfreezing Hubert final projection layer")
                         for param in self.hubert_ser.hubert.project_hid.parameters():
                             param.requires_grad = True
                    if hasattr(self.hubert_ser.hubert, 'feature_projection'):
                        print("  Unfreezing Hubert feature projection layer")
                        for param in self.hubert_ser.hubert.feature_projection.parameters():
                            param.requires_grad = True
                else:
                    print("Warning: Could not find 'layers' attribute in hubert.encoder. Cannot unfreeze specific layers.")
            except AttributeError as e:
                print(f"Warning: Error accessing Hubert layers for unfreezing: {e}. Keeping all backbone layers frozen.")
        else:
             print("Keeping all Hubert backbone parameters frozen.")


    def forward(self, video_input, audio_input_values, audio_attention_mask):
        # video_input shape: [B, T, C, H, W]
        # audio_input_values shape: [B, AudioSeqLen]
        # audio_attention_mask shape: [B, AudioSeqLen]

        # Get Video features
        video_features = self.video_embedder(video_input) # Shape: [B, video_embedding_size]

        # Get Hubert features (before final FC)
        hubert_outputs = self.hubert_ser.hubert(input_values=audio_input_values, attention_mask=audio_attention_mask)
        hubert_features = self.hubert_ser._pool(hubert_outputs.last_hidden_state, audio_attention_mask) # Shape: [B, hubert_feature_dim]

        # Concatenate features
        fused_features = torch.cat((video_features, hubert_features), dim=1)

        # Pass through fusion layer
        fused_output = self.fusion_layer(fused_features)

        # Classify for both tasks
        emotion_logits = self.emotion_head(fused_output)
        laugh_logits = self.humor_head(fused_output).squeeze(-1) # Remove trailing dim for BCEWithLogitsLoss

        return {
            "emotion_logits": emotion_logits,
            "laugh_logits": laugh_logits
        }

# --- Training and Validation Functions ---
def train_epoch(model, dataloader, emotion_criterion, humor_criterion, humor_loss_weight, optimizer, scaler, device, use_amp, add_humor):
    model.train()
    # Ensure parts of Hubert backbone remain in eval mode if *partially* frozen
    if hasattr(model, 'hubert_ser') and hasattr(model.hubert_ser, 'hubert'):
        # Set the main backbone to eval if *any* part is frozen
        # This prevents BatchNorm/Dropout updates in the frozen parts.
        # The un-frozen layers will still compute gradients.
        is_partially_frozen = not all(p.requires_grad for p in model.hubert_ser.hubert.parameters())
        if is_partially_frozen:
             model.hubert_ser.hubert.eval()
             # print("Note: Hubert backbone set to eval() mode because some layers are frozen.") # Reduce verbosity
        else:
             model.hubert_ser.hubert.train() # Ensure it's training if fully unfrozen

    running_loss = 0.0
    running_emotion_loss = 0.0
    running_humor_loss = 0.0
    all_emotion_preds = []
    all_emotion_labels = []
    all_laugh_preds_binary = [] # Binary predictions for accuracy etc.
    all_laugh_labels = []
    progress_bar = tqdm(dataloader, desc="Training Fusion")

    for batch_data in progress_bar:
        # Unpack based on whether humor is added
        if add_humor:
             # HumorDataset yields: video, audio_vals, audio_mask, emotion_label, laugh_label, has_video_mask
             video_data, audio_input_values, audio_attention_mask, emotion_labels, laugh_labels, _ = batch_data # Unpack 6 items
        else:
             # FusionDataset yields: video, audio_vals, audio_mask, emotion_label
             video_data, audio_input_values, audio_attention_mask, emotion_labels = batch_data # Unpack 4 items
             laugh_labels = None # Set to None if not using humor

        video_data = video_data.to(device) if video_data is not None else None
        audio_input_values = audio_input_values.to(device)
        audio_attention_mask = audio_attention_mask.to(device)
        emotion_labels = emotion_labels.to(device)
        if laugh_labels is not None:
            laugh_labels = laugh_labels.to(device).float() # Ensure float for BCE

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            model_outputs = model(video_data, audio_input_values, audio_attention_mask)
            emotion_logits = model_outputs["emotion_logits"]
            laugh_logits = model_outputs["laugh_logits"]

            # Calculate loss
            loss_e = emotion_criterion(emotion_logits, emotion_labels)
            if add_humor and humor_criterion is not None:
                loss_h = humor_criterion(laugh_logits, laugh_labels)
                loss = loss_e + humor_loss_weight * loss_h
                running_humor_loss += loss_h.item()
            else:
                loss = loss_e
                loss_h = torch.tensor(0.0) # Placeholder if humor not active

            running_emotion_loss += loss_e.item()

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

        # Accumulate predictions and labels
        _, emotion_preds = torch.max(emotion_logits, 1)
        all_emotion_preds.extend(emotion_preds.cpu().numpy())
        all_emotion_labels.extend(emotion_labels.cpu().numpy())

        if add_humor:
            laugh_preds_binary = (torch.sigmoid(laugh_logits) >= 0.5).long()
            all_laugh_preds_binary.extend(laugh_preds_binary.cpu().numpy())
            all_laugh_labels.extend(laugh_labels.cpu().numpy().astype(int)) # Ensure labels are int for metrics

        progress_bar.set_postfix(loss=loss.item(), emo_loss=loss_e.item(), hum_loss=loss_h.item())

    epoch_loss = running_loss / len(dataloader)
    epoch_emotion_loss = running_emotion_loss / len(dataloader)
    epoch_emotion_acc = accuracy_score(all_emotion_labels, all_emotion_preds) * 100

    epoch_humor_loss = 0.0
    epoch_humor_acc = 0.0
    if add_humor:
        epoch_humor_loss = running_humor_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        if len(all_laugh_labels) > 0:
             epoch_humor_acc = accuracy_score(all_laugh_labels, all_laugh_preds_binary) * 100

    return epoch_loss, epoch_emotion_loss, epoch_emotion_acc, epoch_humor_loss, epoch_humor_acc


def validate(model, dataloader, emotion_criterion, humor_criterion, humor_loss_weight, device, add_humor):
    model.eval()
    running_loss = 0.0
    running_emotion_loss = 0.0
    running_humor_loss = 0.0
    all_emotion_preds = []
    all_emotion_labels = []
    all_laugh_logits = [] # Store logits for AUROC/PR-AUC
    all_laugh_labels = []
    progress_bar = tqdm(dataloader, desc="Validating Fusion", leave=False)

    with torch.no_grad():
        for batch_data in progress_bar:
            # Unpack based on whether humor is added
            if add_humor:
                 video_data, audio_input_values, audio_attention_mask, emotion_labels, laugh_labels, _ = batch_data
            else:
                 video_data, audio_input_values, audio_attention_mask, emotion_labels = batch_data
                 laugh_labels = None

            video_data = video_data.to(device) if video_data is not None else None
            audio_input_values = audio_input_values.to(device)
            audio_attention_mask = audio_attention_mask.to(device)
            emotion_labels = emotion_labels.to(device)
            if laugh_labels is not None:
                laugh_labels = laugh_labels.to(device).float()

            model_outputs = model(video_data, audio_input_values, audio_attention_mask)
            emotion_logits = model_outputs["emotion_logits"]
            laugh_logits = model_outputs["laugh_logits"]

            # Calculate loss
            loss_e = emotion_criterion(emotion_logits, emotion_labels)
            if add_humor and humor_criterion is not None:
                loss_h = humor_criterion(laugh_logits, laugh_labels)
                loss = loss_e + humor_loss_weight * loss_h
                running_humor_loss += loss_h.item()
            else:
                loss = loss_e
            running_emotion_loss += loss_e.item()
            running_loss += loss.item()

            # Accumulate predictions and labels
            _, emotion_preds = torch.max(emotion_logits, 1)
            all_emotion_preds.extend(emotion_preds.cpu().numpy())
            all_emotion_labels.extend(emotion_labels.cpu().numpy())

            if add_humor:
                all_laugh_logits.extend(laugh_logits.cpu().numpy())
                all_laugh_labels.extend(laugh_labels.cpu().numpy().astype(int)) # Ensure int for metrics

    val_loss = running_loss / len(dataloader)
    val_emotion_loss = running_emotion_loss / len(dataloader)
    val_emotion_acc = accuracy_score(all_emotion_labels, all_emotion_preds) * 100

    print("\nValidation Emotion Classification Report:")
    if SKLEARN_AVAILABLE:
        print(classification_report(all_emotion_labels, all_emotion_preds, target_names=EMOTION_LABELS, zero_division=0))
    else:
        print("  (Requires scikit-learn for detailed report)")

    # Calculate humor metrics
    val_humor_loss = 0.0
    val_humor_metrics = {"acc": 0.0, "auroc": 0.0, "pr_auc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    if add_humor:
        val_humor_loss = running_humor_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        if len(all_laugh_labels) > 0 and len(all_laugh_logits) > 0 and SKLEARN_AVAILABLE:
            print("\nValidation Humor Detection Report:")
            try:
                # Calculate metrics using logits
                val_auroc = roc_auc_score(all_laugh_labels, all_laugh_logits)
                val_pr_auc = average_precision_score(all_laugh_labels, all_laugh_logits)

                # Calculate metrics using binary predictions (e.g., threshold 0.5)
                laugh_preds_binary = (torch.sigmoid(torch.tensor(all_laugh_logits)) >= 0.5).long().numpy()
                val_humor_acc = accuracy_score(all_laugh_labels, laugh_preds_binary) * 100
                precision, recall, f1, _ = precision_recall_fscore_support(all_laugh_labels, laugh_preds_binary, average='binary', zero_division=0)

                print(f"  Accuracy: {val_humor_acc:.2f}%")
                print(f"  AUROC: {val_auroc:.4f}")
                print(f"  PR-AUC: {val_pr_auc:.4f}")
                print(f"  Precision (thresh 0.5): {precision:.4f}")
                print(f"  Recall (thresh 0.5): {recall:.4f}")
                print(f"  F1-Score (thresh 0.5): {f1:.4f}")

                val_humor_metrics = {"acc": val_humor_acc, "auroc": val_auroc, "pr_auc": val_pr_auc, "precision": precision, "recall": recall, "f1": f1}

            except ValueError as e_metric:
                print(f"  Could not calculate some humor metrics: {e_metric}")
                # Assign placeholders if calculation failed
                val_humor_metrics["auroc"] = val_humor_metrics.get("auroc", 0.0)
                val_humor_metrics["pr_auc"] = val_humor_metrics.get("pr_auc", 0.0)
        elif add_humor:
            print("\nValidation Humor Detection Report: Not enough data or scikit-learn not available for metrics.")

    return val_loss, val_emotion_loss, val_emotion_acc, val_humor_loss, val_humor_metrics


# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Train Multimodal Fusion Model")
    # Paths
    parser.add_argument("--manifest_emotion", type=str, required=True, help="Path to main emotion CSV manifest")
    parser.add_argument("--audio_root", type=str, default=None, help="Root directory for audio files (if different from video)")
    parser.add_argument("--video_root", type=str, default=None, help="Root directory for video files (if different from audio)")
    parser.add_argument("--hubert_checkpoint", type=str, required=True, help="Path to the pre-trained Hubert SER model checkpoint (.ckpt)")
    parser.add_argument("--video_checkpoint", type=str, default=None, help="Path to the pre-trained Video (R3D-18) model checkpoint (.pt). Required if --use_video is set.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save fusion models and logs")
    parser.add_argument("--config", type=str, default=None, help="Optional path to YAML config file for detailed settings")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Directory for caching dataset features")

    # Modality Flags
    parser.add_argument("--use_video", action=argparse.BooleanOptionalAction, default=True, help="Include video features in the fusion model (default: True)")

    # Humor Head Specific Arguments
    parser.add_argument("--add_humor", action="store_true", help="Add and train the humor detection head.")
    parser.add_argument("--manifest_humor", type=str, default=None, help="Path to the humor dataset manifest CSV (required if --add_humor).")
    parser.add_argument("--humor_loss_weight", type=float, default=1.0, help="Weight (lambda) for the humor BCE loss (default: 1.0).")
    parser.add_argument("--laugh_threshold", type=float, default=0.5, help="Confidence threshold for calculating laugh precision/recall metrics (default: 0.5).") # Note: Currently calculated in validate

    # Model Hyperparameters
    parser.add_argument("--hubert_model_name", type=str, default="facebook/hubert-base-ls960", help="Name of the Hubert model used for the checkpoint (for feature extractor)")
    parser.add_argument("--hubert_unfreeze_layers", type=int, default=0, help="Number of final Hubert transformer layers to unfreeze (default: 0)")
    parser.add_argument("--fusion_dim", type=int, default=512, help="Dimension of the fusion layer output")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate for fusion/classifier layers")
    parser.add_argument("--num_classes", type=int, default=len(EMOTION_LABELS), help="Number of emotion classes")

    # Training Hyperparameters
    parser.add_argument("--criterion", type=str, default="cross_entropy", choices=["cross_entropy", "focal"], help="Loss function for emotion (default: cross_entropy)")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Gamma parameter for Focal Loss (default: 2.0)")
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine", "plateau"], help="Learning rate scheduler (default: none)")
    parser.add_argument("--cosine_t0", type=int, default=10, help="T_0 for CosineAnnealingWarmRestarts (default: 10)")
    parser.add_argument("--plateau_patience", type=int, default=5, help="Patience for ReduceLROnPlateau (default: 5)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for head/fusion layers")
    parser.add_argument("--lr_backbone", type=float, default=None, help="Optional different learning rate for unfrozen backbone layers")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing factor for Cross Entropy")
    parser.add_argument("--early_stop", type=int, default=10, help="Early stopping patience based on combined metric")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--use_class_weights", action=argparse.BooleanOptionalAction, default=True, help="Use class weights for emotion loss (if applicable)")
    parser.add_argument("--balanced_sampling", action=argparse.BooleanOptionalAction, default=False, help="Use balanced sampling for emotion dataset (ignored if --add_humor)")

    # Data/Loader Settings
    parser.add_argument("--img_size", type=int, default=112, help="Image size for video frames")
    parser.add_argument("--frames", type=int, default=48, help="Number of frames per video clip")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--segment_length", type=float, default=3.0, help="Duration of audio/video segments in seconds")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Target audio sample rate")

    # Augmentation Flags (passed to HumorDataset if used)
    parser.add_argument("--use_spec_augment", action="store_true", help="Apply SpecAugment to audio")
    parser.add_argument("--use_video_augment", action="store_true", help="Apply video augmentations (flip, jitter, etc.)")

    # Feature Flags (passed to HumorDataset if used) - Add defaults matching HumorDataset
    parser.add_argument("--feature_type", type=str, default="raw", help="Audio feature type (raw, mfcc, fbank, hubert, wav2vec2, cnn, etc.)")
    parser.add_argument("--feature_dim", type=int, default=None, help="Dimension of precomputed features (if used)")
    parser.add_argument("--use_hubert", action="store_true", help="Use HuBERT features (extracted within dataset)")
    parser.add_argument("--hubert_model_path", type=str, default="facebook/hubert-base-ls960", help="Path or name of HuBERT model")
    parser.add_argument("--hubert_processor_path", type=str, default=None, help="Path to HuBERT processor (if different from model)")
    parser.add_argument("--hubert_layer", type=int, default=-1, help="HuBERT layer to extract features from")
    # Add other feature flags as needed... (wav2vec2, cnn, facenet, openface, video_features)
    parser.add_argument("--use_wav2vec2", action="store_true", help="Use Wav2Vec2 features")
    parser.add_argument("--wav2vec2_model_path", type=str, default="facebook/wav2vec2-base-960h", help="Path or name of Wav2Vec2 model")
    parser.add_argument("--wav2vec2_processor_path", type=str, default=None, help="Path to Wav2Vec2 processor")
    parser.add_argument("--wav2vec2_layer", type=int, default=-1, help="Wav2Vec2 layer to extract features from")
    parser.add_argument("--use_cnn_features", action="store_true", help="Use precomputed CNN audio features")
    parser.add_argument("--cnn_feature_dir", type=str, default=None, help="Directory of precomputed CNN audio features")
    parser.add_argument("--use_facenet_features", action="store_true", help="Use precomputed FaceNet features")
    parser.add_argument("--facenet_feature_dir", type=str, default=None, help="Directory of precomputed FaceNet features")
    parser.add_argument("--use_openface_features", action="store_true", help="Use precomputed OpenFace features")
    parser.add_argument("--openface_feature_dir", type=str, default=None, help="Directory of precomputed OpenFace features")
    parser.add_argument("--use_video_features", action="store_true", help="Use precomputed video features")
    parser.add_argument("--video_feature_dir", type=str, default=None, help="Directory of precomputed video features")
    parser.add_argument("--video_feature_dim", type=int, default=None, help="Dimension of precomputed video features")
    parser.add_argument("--video_padding_value", type=float, default=0.0, help="Padding value for video features")
    parser.add_argument("--max_video_frames", type=int, default=None, help="Maximum video frames for precomputed features")
    parser.add_argument("--normalize_audio", action="store_true", help="Normalize audio features")
    parser.add_argument("--normalize_video", action="store_true", help="Normalize video features")
    parser.add_argument("--audio_norm_mean_path", type=str, default=None, help="Path to audio mean file")
    parser.add_argument("--audio_norm_std_path", type=str, default=None, help="Path to audio std dev file")
    parser.add_argument("--video_norm_mean_path", type=str, default=None, help="Path to video mean file")
    parser.add_argument("--video_norm_std_path", type=str, default=None, help="Path to video std dev file")
    parser.add_argument("--n_mels", type=int, default=128, help="Number of mel bins for fbank")
    parser.add_argument("--n_fft", type=int, default=400, help="FFT size for fbank")
    parser.add_argument("--hop_length", type=int, default=160, help="Hop length for fbank")
    parser.add_argument("--win_length", type=int, default=400, help="Window length for fbank")


    args = parser.parse_args()

    # --- Argument Validation ---
    if args.add_humor and not args.manifest_humor:
        parser.error("--manifest_humor is required when --add_humor is set.")
    if args.add_humor and not HUMOR_COMPONENTS_AVAILABLE:
         parser.error("HumorDataset or BalancedHumorSampler failed to import. Cannot proceed with --add_humor.")
    if args.use_video and not args.video_checkpoint:
        parser.error("--video_checkpoint is required when --use_video is set.")
    if not args.hubert_checkpoint:
        parser.error("--hubert_checkpoint is required.")
    if not args.use_video:
        print("Warning: --no-use_video was specified, but the current model architecture requires video input. Training will likely fail.")
    # --- End Argument Validation ---

    # Load config if provided, override args if necessary
    config = {}
    if args.config and os.path.exists(args.config):
        print(f"Loading config from {args.config}")
        config = load_yaml(args.config)
        # Example of overriding args with config values (optional)
        # args.lr = config.get('training', {}).get('lr', args.lr)
        # args.batch_size = config.get('training', {}).get('batch_size', args.batch_size)
        # ... etc.

    # Setup device, output dir, etc.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Using device: {device}")

    # --- Create Datasets and Dataloaders ---
    print("Creating Datasets...")
    if args.add_humor:
        print("Using HumorDataset...")
        if not args.manifest_emotion or not args.manifest_humor:
             parser.error("--manifest_emotion and --manifest_humor are required when --add_humor is set.")

        # Instantiate HumorDataset for train and val
        train_dataset = HumorDataset(
            manifest_emotion=args.manifest_emotion,
            manifest_humor=args.manifest_humor,
            split='train',
            audio_root=args.audio_root,
            video_root=args.video_root,
            segment_length=args.segment_length,
            use_video=args.use_video,
            use_spec_augment=args.use_spec_augment,
            use_video_augment=args.use_video_augment,
            sample_rate=args.sample_rate,
            n_mels=args.n_mels, n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.win_length,
            cache_dir=args.cache_dir,
            num_emotion_classes=args.num_classes,
            feature_type=args.feature_type, feature_dim=args.feature_dim,
            use_hubert=args.use_hubert, hubert_model_path=args.hubert_model_path, hubert_processor_path=args.hubert_processor_path, hubert_layer=args.hubert_layer,
            use_wav2vec2=args.use_wav2vec2, wav2vec2_model_path=args.wav2vec2_model_path, wav2vec2_processor_path=args.wav2vec2_processor_path, wav2vec2_layer=args.wav2vec2_layer,
            use_cnn_features=args.use_cnn_features, cnn_feature_dir=args.cnn_feature_dir,
            use_facenet_features=args.use_facenet_features, facenet_feature_dir=args.facenet_feature_dir,
            use_openface_features=args.use_openface_features, openface_feature_dir=args.openface_feature_dir,
            use_video_features=args.use_video_features, video_feature_dir=args.video_feature_dir, video_feature_dim=args.video_feature_dim, video_padding_value=args.video_padding_value, max_video_frames=args.max_video_frames,
            normalize_audio=args.normalize_audio, normalize_video=args.normalize_video,
            audio_norm_mean_path=args.audio_norm_mean_path, audio_norm_std_path=args.audio_norm_std_path,
            video_norm_mean_path=args.video_norm_mean_path, video_norm_std_path=args.video_norm_std_path,
        )
        val_dataset = HumorDataset(
            manifest_emotion=args.manifest_emotion,
            manifest_humor=args.manifest_humor,
            split='val', # Use validation split
            audio_root=args.audio_root,
            video_root=args.video_root,
            segment_length=args.segment_length,
            use_video=args.use_video,
            use_spec_augment=False, # No augmentation for validation
            use_video_augment=False,
            sample_rate=args.sample_rate,
            n_mels=args.n_mels, n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.win_length,
            cache_dir=args.cache_dir,
            num_emotion_classes=args.num_classes,
            feature_type=args.feature_type, feature_dim=args.feature_dim,
            use_hubert=args.use_hubert, hubert_model_path=args.hubert_model_path, hubert_processor_path=args.hubert_processor_path, hubert_layer=args.hubert_layer,
            use_wav2vec2=args.use_wav2vec2, wav2vec2_model_path=args.wav2vec2_model_path, wav2vec2_processor_path=args.wav2vec2_processor_path, wav2vec2_layer=args.wav2vec2_layer,
            use_cnn_features=args.use_cnn_features, cnn_feature_dir=args.cnn_feature_dir,
            use_facenet_features=args.use_facenet_features, facenet_feature_dir=args.facenet_feature_dir,
            use_openface_features=args.use_openface_features, openface_feature_dir=args.openface_feature_dir,
            use_video_features=args.use_video_features, video_feature_dir=args.video_feature_dir, video_feature_dim=args.video_feature_dim, video_padding_value=args.video_padding_value, max_video_frames=args.max_video_frames,
            normalize_audio=args.normalize_audio, normalize_video=args.normalize_video,
            audio_norm_mean_path=args.audio_norm_mean_path, audio_norm_std_path=args.audio_norm_std_path,
            video_norm_mean_path=args.video_norm_mean_path, video_norm_std_path=args.video_norm_std_path,
        )

        # Use BalancedHumorSampler for training
        print("Using BalancedHumorSampler for training.")
        train_batch_sampler = BalancedHumorSampler(train_dataset, batch_size=args.batch_size)
        train_loader = DataLoader(
            train_dataset, batch_sampler=train_batch_sampler, num_workers=args.num_workers,
            pin_memory=True, collate_fn=train_dataset.collate_fn # Use dataset's collate
        )
        # Use default sampler for validation
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
            pin_memory=True, collate_fn=val_dataset.collate_fn # Use dataset's collate
        )

    else:
        # Use original FusionDataset
        print("Using FusionDataset (Emotion only)...")
        train_dataset = FusionDataset(
            manifest_file=args.manifest_emotion,
            audio_base_path=args.audio_root or args.video_root, # Assume one root if audio_root not given
            use_video=args.use_video,
            hubert_model_name=args.hubert_model_name,
            split='train',
            frames=args.frames,
            img_size=args.img_size,
            config=config,
            augment=True
        )
        val_dataset = FusionDataset(
            manifest_file=args.manifest_emotion,
            audio_base_path=args.audio_root or args.video_root,
            use_video=args.use_video,
            hubert_model_name=args.hubert_model_name,
            split='val',
            frames=args.frames,
            img_size=args.img_size,
            config=config,
            augment=False
        )

        # Optional balanced sampling for emotion-only training
        train_sampler = None
        shuffle_train = True
        if args.balanced_sampling:
             print("Using WeightedRandomSampler for balanced emotion training.")
             # Calculate weights for WeightedRandomSampler
             labels = train_dataset.data['manifest_label_idx'].tolist()
             class_counts = Counter(labels)
             class_weights_sample = [1.0 / class_counts[i] for i in range(args.num_classes)]
             sample_weights = [class_weights_sample[label] for label in labels]
             train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
             shuffle_train = False # Sampler handles shuffling

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=train_sampler, shuffle=shuffle_train,
            num_workers=args.num_workers, pin_memory=True, collate_fn=train_dataset.collate_fn
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, collate_fn=val_dataset.collate_fn
        )

    print("Datasets and Dataloaders created.")

    # --- Calculate Class Weights for Emotion Loss (if needed) ---
    emotion_class_weights = None
    if args.use_class_weights and args.criterion == "focal":
        print("Calculating class weights for Emotion Focal Loss...")
        try:
            # Get emotion labels from the training dataset
            if args.add_humor:
                # HumorDataset should provide a way to get emotion labels
                if hasattr(train_dataset, 'get_emotion_labels'):
                    all_emotion_labels = train_dataset.get_emotion_labels()
                else:
                    print("Warning: HumorDataset does not have 'get_emotion_labels' method. Cannot calculate emotion weights.")
                    all_emotion_labels = []
            else:
                # FusionDataset stores labels in 'manifest_label_idx'
                if hasattr(train_dataset, 'data') and 'manifest_label_idx' in train_dataset.data.columns:
                    all_emotion_labels = train_dataset.data['manifest_label_idx'].tolist()
                else:
                     print("Warning: Could not find emotion labels in FusionDataset. Cannot calculate weights.")
                     all_emotion_labels = []

            if all_emotion_labels:
                label_counts = Counter(all_emotion_labels)
                total_samples = len(all_emotion_labels)
                weights = []
                for i in range(args.num_classes): # Use args.num_classes
                    count = label_counts.get(i, 0)
                    weight = total_samples / (args.num_classes * count) if count > 0 else 1.0
                    weights.append(weight)
                emotion_class_weights = torch.tensor(weights, dtype=torch.float).to(device)
                print(f"Emotion class weights for Focal Loss: {emotion_class_weights}")
            else:
                 print("No emotion labels found to calculate weights.")

        except Exception as e:
             print(f"Error calculating emotion class weights: {e}")
             traceback.print_exc()
             emotion_class_weights = None # Fallback

    elif args.use_class_weights and args.criterion == "cross_entropy":
         print("Calculating class weights for Emotion Cross Entropy Loss...")
         # Similar logic as above, but weights are used differently by CE
         try:
            if args.add_humor:
                if hasattr(train_dataset, 'get_emotion_labels'):
                    all_emotion_labels = train_dataset.get_emotion_labels()
                else:
                    print("Warning: HumorDataset does not have 'get_emotion_labels' method. Cannot calculate emotion weights.")
                    all_emotion_labels = []
            else:
                if hasattr(train_dataset, 'data') and 'manifest_label_idx' in train_dataset.data.columns:
                    all_emotion_labels = train_dataset.data['manifest_label_idx'].tolist()
                else:
                     print("Warning: Could not find emotion labels in FusionDataset. Cannot calculate weights.")
                     all_emotion_labels = []

            if all_emotion_labels:
                label_counts = Counter(all_emotion_labels)
                total_samples = len(all_emotion_labels)
                weights = []
                for i in range(args.num_classes):
                    count = label_counts.get(i, 0)
                    # Weight calculation might differ for CE depending on library implementation,
                    # often inverse frequency is used directly.
                    weight = total_samples / (args.num_classes * count) if count > 0 else 1.0
                    weights.append(weight)
                emotion_class_weights = torch.tensor(weights, dtype=torch.float).to(device)
                print(f"Emotion class weights for Cross Entropy: {emotion_class_weights}")
            else:
                 print("No emotion labels found to calculate weights.")

         except Exception as e:
             print(f"Error calculating emotion class weights: {e}")
             traceback.print_exc()
             emotion_class_weights = None # Fallback
    else:
        print("Not using class weights for emotion loss.")


    # --- Create Model ---
    print("Creating Fusion Model...")
    model = FusionModel(
        num_classes=args.num_classes,
        video_checkpoint_path=args.video_checkpoint,
        hubert_checkpoint_path=args.hubert_checkpoint,
        hubert_model_name=args.hubert_model_name,
        fusion_dim=args.fusion_dim,
        dropout=args.dropout,
        hubert_unfreeze_layers=args.hubert_unfreeze_layers # Pass unfreeze count
    ).to(device)
    print("Model created.")

    # --- Optimizer, Scheduler, Loss ---
    print("Setting up optimizer, scheduler, and loss...")
    optimizer_params = []
    if args.hubert_unfreeze_layers > 0 and args.lr_backbone is not None:
        print(f"Using differential learning rate: Backbone LR={args.lr_backbone}, Head LR={args.lr}")
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Check if param belongs to the unfrozen Hubert layers or related components
                is_hubert_backbone = False
                if name.startswith("hubert_ser.hubert"):
                    # More specific check for unfrozen layers might be needed if only *some* are unfrozen
                    # For simplicity now, assume all params starting with hubert_ser.hubert are backbone if unfreezing > 0
                    is_hubert_backbone = True

                if is_hubert_backbone:
                    backbone_params.append(param)
                else:
                    head_params.append(param) # Fusion, emotion head, humor head, video head (if unfrozen)

        print(f"  Backbone params ({len(backbone_params)} tensors) LR: {args.lr_backbone}")
        print(f"  Head params ({len(head_params)} tensors) LR: {args.lr}")
        optimizer_params = [
            {'params': backbone_params, 'lr': args.lr_backbone},
            {'params': head_params, 'lr': args.lr}
        ]
        # Check if any parameters were missed
        total_grad_params = sum(1 for p in model.parameters() if p.requires_grad)
        assigned_params = len(backbone_params) + len(head_params)
        if total_grad_params != assigned_params:
             print(f"Warning: Mismatch in assigning optimizer params! Total grad params: {total_grad_params}, Assigned: {assigned_params}")

    else:
        print(f"Using single learning rate: {args.lr}")
        optimizer_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler
    scheduler = None
    if args.scheduler == "cosine":
        print(f"Using CosineAnnealingWarmRestarts scheduler with T_0={args.cosine_t0}")
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.cosine_t0, eta_min=args.lr * 0.01)
    elif args.scheduler == "plateau":
        print(f"Using ReduceLROnPlateau scheduler with patience={args.plateau_patience}")
        # Monitor combined metric: val_emotion_acc + val_humor_auroc (or just emotion if humor not added)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.plateau_patience, verbose=True)
    else:
        print("No learning rate scheduler selected.")

    # Loss Function
    if args.criterion == "focal":
        print(f"Using Focal Loss for emotion with gamma={args.focal_gamma} and alpha={emotion_class_weights is not None}")
        emotion_criterion = FocalLoss(alpha=emotion_class_weights, gamma=args.focal_gamma)
    else:
        print(f"Using Cross Entropy Loss for emotion with label smoothing={args.label_smoothing} and weights={emotion_class_weights is not None}")
        emotion_criterion = nn.CrossEntropyLoss(weight=emotion_class_weights, label_smoothing=args.label_smoothing)

    humor_criterion = None
    if args.add_humor:
        print("Using BCEWithLogitsLoss for humor detection.")
        humor_criterion = nn.BCEWithLogitsLoss()

    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None

    # --- Training Loop ---
    best_combined_metric = -1.0 # Initialize for combined metric tracking
    patience_counter = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"fusion_model_{timestamp}"

    print("Starting training loop...")
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        train_loss, train_emo_loss, train_emo_acc, train_hum_loss, train_hum_acc = train_epoch(
            model, train_loader, emotion_criterion, humor_criterion, args.humor_loss_weight,
            optimizer, scaler, device, args.fp16, args.add_humor
        )
        val_loss, val_emo_loss, val_emo_acc, val_hum_loss, val_humor_metrics = validate(
            model, val_loader, emotion_criterion, humor_criterion, args.humor_loss_weight,
            device, args.add_humor
        )

        # Log metrics
        log_str = (f"Epoch {epoch}: Train Loss={train_loss:.4f} (Emo: {train_emo_loss:.4f}, Hum: {train_hum_loss:.4f}), "
                   f"Train Acc (Emo: {train_emo_acc:.2f}%, Hum: {train_hum_acc:.2f}%) | "
                   f"Val Loss={val_loss:.4f} (Emo: {val_emo_loss:.4f}, Hum: {val_hum_loss:.4f}), "
                   f"Val Acc (Emo: {val_emo_acc:.2f}%, Hum: {val_humor_metrics.get('acc', 0.0):.2f}%)")
        if args.add_humor:
            log_str += f" | Val Humor (AUROC: {val_humor_metrics.get('auroc', 0.0):.4f}, PR-AUC: {val_humor_metrics.get('pr_auc', 0.0):.4f})"
        print(log_str)

        # --- Early Stopping and Best Model Saving ---
        # Define combined metric for early stopping and saving best model
        # Example: Weighted sum of Val Emotion Acc and Val Humor AUROC
        current_combined_metric = val_emo_acc
        if args.add_humor:
             # Give equal weight for simplicity, adjust as needed
             current_combined_metric = (val_emo_acc + val_humor_metrics.get('auroc', 0.0) * 100) / 2

        if current_combined_metric > best_combined_metric:
            best_combined_metric = current_combined_metric
            patience_counter = 0
            print(f"Saving best model with combined metric: {best_combined_metric:.4f} (Val Emo Acc: {val_emo_acc:.2f}%)")
            save_path = os.path.join(args.output_dir, f"{model_name}_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_combined_metric': best_combined_metric,
                'val_emotion_acc': val_emo_acc,
                'val_humor_metrics': val_humor_metrics,
                'args': args # Save args for reproducibility
            }, save_path)
        else:
            patience_counter += 1
            print(f"Combined metric did not improve. Patience: {patience_counter}/{args.early_stop}")
            if patience_counter >= args.early_stop:
                print(f"Early stopping triggered after {args.early_stop} epochs without improvement on combined validation metric.")
                break

        # Step the scheduler
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(current_combined_metric) # Step based on the combined metric
            else:
                scheduler.step() # Others step per epoch

    print(f"\nTraining finished. Best Combined Metric: {best_combined_metric:.4f}")
    print(f"Best model saved to: {os.path.join(args.output_dir, f'{model_name}_best.pt')}")

if __name__ == "__main__":
    main()
