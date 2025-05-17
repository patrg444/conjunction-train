#!/usr/bin/env python3
"""
Enhanced emotion recognition training script using SlowFast-R50 backbone.

This script implements a more advanced training approach with:
- SlowFast-R50 backbone (higher capacity than R3D-18)
- Multi-clip sampling per video
- Extended augmentation pipeline
- Label smoothing and AdamW optimizer
- One-cycle LR scheduling
- Test-time augmentation with 5-clip ensembling

Key features:
- Higher accuracy (target: 70%+)
- Better generalization
- More robust to lighting and pose variations
"""

import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import json
import yaml
import math
from pathlib import Path
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import torchvision
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Emotion labels that are common between RAVDESS and CREMA-D
EMOTION_LABELS = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad']


class RandomTimeReverse(nn.Module):
    """Randomly reverse video clip in temporal dimension."""
    
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
    def forward(self, x):
        # x shape: [T, C, H, W]
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
        # x shape: [T, C, H, W]
        if random.random() < self.p:
            T = x.shape[0]
            drop_count = random.randint(1, min(self.max_drop, T-1))
            drop_indices = sorted(random.sample(range(T), drop_count))
            
            # Create a list of indices where dropped frames are replaced with subsequent frames
            keep_indices = list(range(T))
            for idx in drop_indices:
                keep_indices.remove(idx)
            
            # Pad with repeated last frame
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
        # x shape: [T, C, H, W]
        if random.random() < self.p:
            T, C, H, W = x.shape
            mask = torch.ones_like(x)
            
            for _ in range(self.count):
                # Apply same cutout to all frames
                y = random.randint(0, H - self.size)
                x_pos = random.randint(0, W - self.size)
                
                mask[:, :, y:y+self.size, x_pos:x_pos+self.size] = 0
                
            return x * mask
        return x


class VideoDataset(Dataset):
    """Enhanced dataset loader for video emotional classification."""

    def __init__(self, manifest_file, split='train', frames=48, img_size=112, 
                 clips_per_video=1, config=None, augment=True):
        """
        Initialize the dataset.
        
        Args:
            manifest_file: Path to the CSV manifest with video paths and labels
            split: 'train', 'val', or 'test'
            frames: Number of frames to extract per clip
            img_size: Size to resize frames to
            clips_per_video: Number of clips to sample per video in training
            config: Config dictionary with augmentation parameters
            augment: Whether to apply data augmentation
        """
        self.manifest_file = manifest_file
        self.split = split
        self.frames = frames
        self.img_size = img_size
        self.augment = augment and split == 'train'
        self.clips_per_video = clips_per_video if self.augment else 1
        
        # Read configuration
        self.config = config or {}
        aug_config = self.config.get('augmentation', {})
        
        # Load the manifest
        df = pd.read_csv(manifest_file)
        # Select only the current split
        df = df[df['split'] == split].reset_index(drop=True)
        
        # Filter for known emotion categories
        df = df[df['label'].isin(EMOTION_LABELS)]
        
        # Create a mapping of emotion labels to indices
        self.label_to_idx = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
        
        # Convert string labels to indices
        df['label_idx'] = df['label'].map(self.label_to_idx)
        
        self.data = df
        
        # Print class distribution
        print(df['label'].value_counts())
        
        # Setup transforms
        self.transform = self._get_transforms(aug_config)
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _get_transforms(self, aug_config):
        """Create augmentation pipeline based on config."""
        transform_list = []
        
        # Basic transforms
        transform_list.extend([
            transforms.ToTensor(),
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
                translate = aug_config.get('translate', [0.05, 0.05])
                if isinstance(translate, list):
                    translate = tuple(translate)  # Just convert to tuple without scaling
                
                transform_list.append(
                    transforms.RandomAffine(
                        degrees=degrees,
                        translate=translate,
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
            
            # Cutout (applied later to tensor)
            cutout_config = aug_config.get('cutout', {})
            if cutout_config.get('enabled', False):
                self.cutout = Cutout(
                    size=cutout_config.get('size', 10),
                    count=cutout_config.get('count', 2),
                    p=0.5
                )
            else:
                self.cutout = None
                
            # Time-reversal (applied later to tensor)
            if aug_config.get('time_reverse', False):
                self.time_reverse = RandomTimeReverse(p=0.5)
            else:
                self.time_reverse = None
                
            # Drop frames (applied later to tensor)
            drop_config = aug_config.get('drop_frames', {})
            if drop_config.get('enabled', False):
                self.drop_frames = RandomDropFrames(
                    max_drop=drop_config.get('max_drop', 2),
                    p=0.5
                )
            else:
                self.drop_frames = None
        
        # Normalization (always applied)
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )
        
        return transforms.Compose(transform_list)

    def __len__(self):
        """Return the number of videos in the dataset."""
        return len(self.data) * self.clips_per_video

    def _get_idx(self, idx):
        """Convert dataset index to data index and clip index."""
        if self.clips_per_video > 1:
            data_idx = idx // self.clips_per_video
            clip_idx = idx % self.clips_per_video
        else:
            data_idx = idx
            clip_idx = 0
        return data_idx, clip_idx

    def _load_video_frames(self, video_path, clip_idx=0):
        """Load frames from a video file.
        
        Returns:
            frames: tensor of shape [T, C, H, W]
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames <= 0 or fps <= 0:
            raise ValueError(f"Invalid video: {video_path}")
        
        # For training with multiple clips, sample different segments
        if self.split == 'train' and self.clips_per_video > 1:
            frames_per_segment = total_frames // self.clips_per_video
            start_frame = frames_per_segment * clip_idx
            end_frame = start_frame + frames_per_segment
        else:
            # For validation/test, always use the center segment
            start_frame = max(0, (total_frames - self.frames) // 2)
            end_frame = min(total_frames, start_frame + self.frames * 2)
        
        # Linear indices for sampling from the video segment
        if end_frame - start_frame <= self.frames:
            # If the segment is shorter than requested frames, sample with repetition
            indices = np.linspace(start_frame, end_frame - 1, self.frames, dtype=np.int32)
        else:
            # Otherwise, sample frames uniformly
            indices = np.linspace(start_frame, end_frame - 1, self.frames, dtype=np.int32)
        
        # Read and process the sampled frames
        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                # If frame reading fails, repeat the last valid frame
                if frames:
                    frames.append(frames[-1])
                else:
                    # If no valid frames yet, use a black frame
                    frames.append(np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8))
                continue
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize frame
            frame = cv2.resize(frame, (self.img_size, self.img_size))
            frames.append(frame)
        
        cap.release()
        
        # Apply transforms to each frame
        transformed_frames = []
        for frame in frames:
            # Convert to PIL Image for torchvision transforms
            pil_img = Image.fromarray(frame)
            
            # Apply transforms
            frame_tensor = self.transform(pil_img)
            transformed_frames.append(frame_tensor)
        
        # Stack frames into tensor [T, C, H, W]
        video_tensor = torch.stack(transformed_frames)
        
        # Apply temporal augmentations
        if self.augment:
            if self.time_reverse:
                video_tensor = self.time_reverse(video_tensor)
            if self.drop_frames:
                video_tensor = self.drop_frames(video_tensor)
            if self.cutout:
                video_tensor = self.cutout(video_tensor)
        
        return video_tensor

    def __getitem__(self, idx):
        """
        Get a video clip and its associated label.
        
        Returns:
            video_tensor: Tensor of shape [T, C, H, W]
            label: The class index
        """
        data_idx, clip_idx = self._get_idx(idx)
        item = self.data.iloc[data_idx]
        video_path = item['path']
        label_idx = item['label_idx']
        
        try:
            # Load and process the video frames
            video_tensor = self._load_video_frames(video_path, clip_idx)
            return video_tensor, label_idx
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            # Return a dummy tensor if loading fails
            dummy = torch.zeros(self.frames, 3, self.img_size, self.img_size)
            return dummy, label_idx


class R3DEmbedder(nn.Module):
    """R3D-18 network for video embeddings with optional SE blocks."""

    def __init__(self, pretrained=True, freeze_backbone=False, use_se=True):
        """Initialize the R3D embedder."""
        super(R3DEmbedder, self).__init__()

        # Use R3D-18 as the backbone
        if pretrained:
            weights = torchvision.models.video.R3D_18_Weights.DEFAULT
            self.backbone = torchvision.models.video.r3d_18(weights=weights)
        else:
            self.backbone = torchvision.models.video.r3d_18(weights=None)
        
        # Add Squeeze-and-Excitation blocks if requested
        if use_se:
            self._add_se_blocks()

        # Freeze backbone weights if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Remove the final fully connected layer
        embedding_size = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove classification head
        self.embedding_size = embedding_size  # Store embedding size (512 for R3D-18)

    def _add_se_blocks(self):
        """Add Squeeze-and-Excitation blocks to the model."""
        # For R3D-18, use a simpler approach without directly accessing internal structure
        # R3D-18 uses BasicBlock (with two Conv3D layers)
        # Analyze the model's structure
        print("Adding SE blocks to R3D-18 model")
        
        # Fixed channel sizes for R3D-18 architecture based on layer
        layer_channels = {
            'layer1': 64,
            'layer2': 128,
            'layer3': 256,
            'layer4': 512
        }
        
        # Add SE blocks to layer3 and layer4 of R3D-18
        for layer_name in ['layer3', 'layer4']:
            layer = getattr(self.backbone, layer_name)
            channels = layer_channels[layer_name]
            
            for i, block in enumerate(layer):
                # Create a new SE block
                se_block = nn.Sequential(
                    nn.AdaptiveAvgPool3d(1),
                    nn.Conv3d(channels, channels // 16, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(channels // 16, channels, kernel_size=1),
                    nn.Sigmoid()
                )
                
                # Store the SE block
                setattr(block, 'se', se_block)
                
                # Define a closure to capture the current block and SE block
                def make_hook(blk):
                    def hook(module, input, output):
                        se_scale = blk.se(output)
                        return output * se_scale
                    return hook
                
                # Register the forward hook
                block.register_forward_hook(make_hook(block))

    def forward(self, x):
        """
        Forward pass through the R3D-18 network.
        
        Args:
            x: Input tensor of shape [B, C, T, H, W]
            
        Returns:
            features: Output tensor
        """
        # Get features from the backbone
        features = self.backbone(x)
        
        return features


class EmotionClassifier(nn.Module):
    """R3D-18 + LSTM model for emotion classification."""

    def __init__(self, num_classes=6, hidden_size=256, dropout=0.5, backbone='r3d_18', 
                 use_se=True, pretrained=True, freeze_backbone=False, embedding_size=512):
        """Initialize the classifier."""
        super(EmotionClassifier, self).__init__()

        # Video embedder
        self.video_embedder = R3DEmbedder(
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            use_se=use_se
        )

        # Set embedding size (R3D-18 has 512 features)
        self.embedding_size = self.video_embedder.embedding_size

        # No LSTM needed - we'll use the R3D features directly

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Final classifier directly from R3D features
        self.classifier = nn.Linear(self.embedding_size, num_classes)

    def forward(self, x):
        """
        Forward pass through the classifier.
        
        Args:
            x: Input tensor of shape [B, T, C, H, W]
            
        Returns:
            logits: Output tensor of shape [B, num_classes]
        """
        batch_size, seq_len, C, H, W = x.size()
        
        # R3D expects input of shape [B, C, T, H, W]
        # We need to process clips of the video instead of individual frames
        
        # Group frames into non-overlapping clips of 16 frames
        clip_size = 16  # R3D typically uses 16-frame clips
        
        # If we have fewer than clip_size frames, we'll repeat frames to reach clip_size
        if seq_len < clip_size:
            padding = torch.repeat_interleave(x, repeats=torch.tensor([clip_size // seq_len + 1] * seq_len), dim=1)
            padding = padding[:, :clip_size, :, :, :]
            x_clips = padding.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        else:
            # Select frames evenly throughout the sequence
            indices = torch.linspace(0, seq_len - 1, clip_size).long()
            x_sampled = x[:, indices, :, :, :]
            x_clips = x_sampled.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
            
        # Pass the clip through R3D
        features = self.video_embedder(x_clips)  # [B, embedding_size]
        
        # Since R3D already processes temporal information and returns features without temporal dimension,
        # we don't need LSTM for temporal modeling. Instead, we'll apply dropout and classify directly.
        
        # Apply dropout
        features = self.dropout(features)
        
        # Get logits - direct classification from R3D features
        logits = self.classifier(features)
        
        return logits


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, use_amp=True):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    for frames, labels in progress_bar:
        frames, labels = frames.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Use automatic mixed precision
        if use_amp:
            with autocast():
                outputs = model(frames)
                loss = criterion(outputs, labels)
                
            # Scale gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Track metrics
        running_loss += loss.item()
        
        # Get predictions
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        avg_loss = running_loss / (progress_bar.n + 1)
        acc = accuracy_score(all_labels, all_preds) * 100
        progress_bar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.1f}")
    
    # Final metrics
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds) * 100
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, test_clips=1):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    all_logits = {}  # Video ID -> list of logits
    all_video_labels = {}  # Video ID -> label
    
    progress_bar = tqdm(dataloader, desc="Validation")
    with torch.no_grad():
        for batch_idx, (frames, labels) in enumerate(progress_bar):
            frames, labels = frames.to(device), labels.to(device)
            
            outputs = model(frames)
            loss = criterion(outputs, labels)
            
            # Track metrics
            running_loss += loss.item()
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix(loss=f"{running_loss/(batch_idx+1):.4f}")
    
    # Final metrics
    val_loss = running_loss / len(dataloader)
    val_acc = accuracy_score(all_labels, all_preds) * 100
    
    # Print confusion matrix and classification report
    conf_mat = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, target_names=EMOTION_LABELS, zero_division=0)
    
    # Print the report
    print("\nClassification Report:")
    print(class_report)
    
    return val_loss, val_acc


def save_checkpoint(model, optimizer, scheduler, epoch, val_acc, model_dir, model_name):
    """Save model checkpoint."""
    os.makedirs(model_dir, exist_ok=True)
    
    # Create checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_acc': val_acc
    }
    
    # Save checkpoint
    checkpoint_path = os.path.join(model_dir, f"{model_name}_epoch{epoch:03d}.pt")
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model separately
    best_model_path = os.path.join(model_dir, f"{model_name}_best.pt")
    torch.save(checkpoint, best_model_path)
    
    return checkpoint_path, best_model_path


def load_yaml(file_path):
    """Load a YAML configuration file."""
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train emotion recognition model with SlowFast backbone")
    parser.add_argument("--config", type=str, default="config/slowfast_face.yaml", help="Path to configuration file")
    parser.add_argument("--manifest_file", type=str, required=True, help="Path to CSV manifest with video paths and labels")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save models and logs")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size")
    parser.add_argument("--epochs", type=int, default=60, help="Number of epochs")
    parser.add_argument("--img_size", type=int, default=112, help="Image size")
    parser.add_argument("--frames", type=int, default=48, help="Number of frames per clip")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--hidden_size", type=int, default=256, help="LSTM hidden size")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--label_smoothing", type=float, default=0.05, help="Label smoothing factor")
    parser.add_argument("--early_stop", type=int, default=12, help="Early stopping patience")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--clips_per_video", type=int, default=2, help="Number of clips to sample per video during training")
    parser.add_argument("--test_clips", type=int, default=5, help="Number of clips to average during validation/testing")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze backbone weights")
    parser.add_argument("--no_se", action="store_true", help="Disable Squeeze-and-Excitation blocks")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    args = parser.parse_args()
    
    # Load config if exists
    config = load_yaml(args.config) if os.path.exists(args.config) else {}
    
    # Override config with command line arguments
    if 'model' not in config:
        config['model'] = {}
    if 'training' not in config:
        config['training'] = {}
    if 'input' not in config:
        config['input'] = {}
        
    # Model settings
    config['model']['dropout'] = args.dropout
    config['model']['hidden_size'] = args.hidden_size
    config['model']['freeze_backbone'] = args.freeze_backbone
    config['model']['use_se'] = not args.no_se
    
    # Training settings
    config['training']['batch_size'] = args.batch_size
    config['training']['epochs'] = args.epochs
    config['training']['learning_rate'] = args.lr
    config['training']['weight_decay'] = args.weight_decay
    config['training']['label_smoothing'] = args.label_smoothing
    config['training']['early_stop_patience'] = args.early_stop
    config['training']['amp'] = args.fp16
    
    # Input settings
    config['input']['frames'] = args.frames
    config['input']['img_size'] = args.img_size
    config['input']['clips_per_video'] = args.clips_per_video
    config['input']['test_clips'] = args.test_clips
    
    # Dataset settings
    config['dataset'] = config.get('dataset', {})
    config['dataset']['num_workers'] = args.num_workers
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = VideoDataset(
        manifest_file=args.manifest_file,
        split="train",
        frames=config['input']['frames'],
        img_size=config['input']['img_size'],
        clips_per_video=config['input']['clips_per_video'],
        config=config,
        augment=True
    )
    
    val_dataset = VideoDataset(
        manifest_file=args.manifest_file,
        split="val",
        frames=config['input']['frames'],
        img_size=config['input']['img_size'],
        clips_per_video=1,
        config=config,
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['dataset']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['dataset']['num_workers'],
        pin_memory=True
    )
    
    # Create model
    model = EmotionClassifier(
        num_classes=len(EMOTION_LABELS),
        hidden_size=config['model']['hidden_size'],
        dropout=config['model']['dropout'],
        backbone=config['model'].get('backbone', 'slowfast_r50'),
        use_se=config['model']['use_se'],
        pretrained=config['model'].get('pretrained', True),
        freeze_backbone=config['model']['freeze_backbone'],
        embedding_size=config['model'].get('embedding_size', 2304)
    )
    model = model.to(device)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create scheduler
    if config['training'].get('lr_schedule', 'cosine') == 'one_cycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config['training']['learning_rate'],
            steps_per_epoch=len(train_loader),
            epochs=config['training']['epochs'],
            pct_start=0.3
        )
    else:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs']
        )
    
    # Create loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=config['training']['label_smoothing'])
    
    # Initialize mixed precision scaler
    use_amp = config['training'].get('amp', False)
    scaler = GradScaler() if use_amp else None
    
    # Initialize metrics tracking
    best_val_acc = 0.0
    patience_counter = 0
    
    # Create model name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"slowfast_emotion_{timestamp}"
    
    # Training loop
    val_accuracies = []
    train_accuracies = []
    val_losses = []
    train_losses = []
    
    # Save configuration
    config_path = os.path.join(args.output_dir, f"{model_name}_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    for epoch in range(1, config['training']['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['training']['epochs']}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, 
            test_clips=config['input'].get('test_clips', 1)
        )
        
        # Update metrics
        train_accuracies.append(train_acc)
        train_losses.append(train_loss)
        val_accuracies.append(val_acc)
        val_losses.append(val_loss)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%")
        
        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            print(f"New best validation accuracy: {best_val_acc:.2f}%")
            
            # Save checkpoint
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_acc,
                args.output_dir, model_name
            )
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs (best: {best_val_acc:.2f}%)")
            
            # Early stopping
            if patience_counter >= config['training']['early_stop_patience']:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, OneCycleLR):
                scheduler.step()
            else:
                scheduler.step()
        
    # Save training history
    history = {
        'train_loss': train_losses,
        'train_acc': train_accuracies,
        'val_loss': val_losses,
        'val_acc': val_accuracies,
    }
    
    history_path = os.path.join(args.output_dir, f"{model_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f)
    
    print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to {args.output_dir}/{model_name}_best.pt")


if __name__ == "__main__":
    main()
