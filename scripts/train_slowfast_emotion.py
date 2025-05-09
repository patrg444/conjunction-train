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
# Note: These are kept for potential reuse but not used for smile detection
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
    """Enhanced dataset loader for video classification (adapted for smile)."""

    def __init__(self, manifest_file, split='train', frames=48, img_size=112, 
                 clips_per_video=1, config=None, augment=True):
        """
        Initialize the dataset.
        
        Args:
            manifest_file: Path to the CSV manifest (e.g., celeba_smile_train.csv)
            split: 'train', 'val', or 'test' (used mainly for augmentation flag)
            frames: Number of frames to extract per clip
            img_size: Size to resize frames to
            clips_per_video: Number of clips to sample per video in training
            config: Config dictionary (used for DATA.PATH_PREFIX)
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
        aug_config = self.config.get('augmentation', {}) # Check if augmentation section exists in config
        
        # Load the manifest
        df = pd.read_csv(manifest_file)
        
        # --- Modifications for CelebA Smile Manifest ---
        # The following lines related to 'split' and 'label' filtering are removed
        # as they are not needed when using separate manifest files like celeba_smile_*.csv
        if 'smile_label' not in df.columns:
             raise ValueError(f"Manifest file {manifest_file} must contain a 'smile_label' column.")
        df['label_idx'] = df['smile_label'] # Use the smile_label column directly
        # --- End Modifications ---

        self.data = df

        # Print class distribution for smile_label
        print(f"Smile label distribution for {split} split ({os.path.basename(manifest_file)}):")
        print(df['smile_label'].value_counts())
        
        # Setup transforms
        self.transform = self._get_transforms(aug_config)

    def _get_transforms(self, aug_config):
        """Create augmentation pipeline based on config."""
        transform_list = []
        
        # --- Note: For CelebA (images), many video augmentations might not apply directly ---
        # --- We'll keep them but they might need adjustment if using image inputs ---
        
        # Basic resize is handled in _load_video_frames
        # ToTensor is applied after augmentations

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
                transform_list.append(
                    transforms.RandomAffine(
                        degrees=degrees,
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
            
            # --- Temporal augmentations (applied later if needed, less relevant for single images) ---
            self.cutout = None
            self.time_reverse = None
            self.drop_frames = None
            
            cutout_config = aug_config.get('cutout', {})
            if cutout_config.get('enabled', False):
                 self.cutout = Cutout(
                    size=cutout_config.get('size', 10),
                    count=cutout_config.get('count', 2),
                    p=0.5
                 )
                 
            if aug_config.get('time_reverse', False):
                 self.time_reverse = RandomTimeReverse(p=0.5)
                 
            drop_config = aug_config.get('drop_frames', {})
            if drop_config.get('enabled', False):
                 self.drop_frames = RandomDropFrames(
                    max_drop=drop_config.get('max_drop', 2),
                    p=0.5
                 )
        else:
             self.cutout = None
             self.time_reverse = None
             self.drop_frames = None

        # Normalization is applied after ToTensor in _load_video_frames
        
        return transforms.Compose(transform_list)

    def __len__(self):
        """Return the number of videos/images in the dataset."""
        return len(self.data) 

    def _get_idx(self, idx):
        """Convert dataset index to data index and clip index."""
        data_idx = idx
        clip_idx = 0 # Not relevant for single images
        return data_idx, clip_idx

    def _load_video_frames(self, image_path, clip_idx=0):
        """Load frames (or single image repeated) for SlowFast."""
        
        # --- Adaptation for Single Image Input (CelebA) ---
        try:
            # Load the single image
            frame = cv2.imread(image_path)
            if frame is None:
                raise IOError(f"Could not read image file: {image_path}")
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.img_size, self.img_size))
            pil_img = Image.fromarray(frame)
            augmented_pil_img = self.transform(pil_img) 
            frame_tensor = TF.to_tensor(augmented_pil_img)
            normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            frame_tensor = normalizer(frame_tensor)
            video_tensor = frame_tensor.unsqueeze(1).repeat(1, self.frames, 1, 1) # [C, T, H, W]
            video_tensor_t_first = video_tensor.permute(1, 0, 2, 3) # [T, C, H, W]
            if self.augment:
                if self.time_reverse:
                    video_tensor_t_first = self.time_reverse(video_tensor_t_first)
                if self.cutout:
                     video_tensor_t_first = self.cutout(video_tensor_t_first) 
            final_video_tensor = video_tensor_t_first.permute(1, 0, 2, 3) # [C, T, H, W]
            return final_video_tensor

        except Exception as e:
             print(f"Error processing image {image_path}: {e}", file=sys.stderr)
             return torch.zeros(3, self.frames, self.img_size, self.img_size)


    def __getitem__(self, idx):
        """
        Get a video clip (repeated image) and its associated label.
        
        Returns:
            video_tensor: Tensor of shape [C, T, H, W] (SlowFast input format)
            label: The class index (smile_label)
        """
        data_idx, clip_idx = self._get_idx(idx) 
        item = self.data.iloc[data_idx]
        
        base_path = self.config.get('DATA', {}).get('PATH_PREFIX', '/home/ubuntu/datasets/') 
        relative_video_path = item['rel_video'] 
        image_path = os.path.join(base_path, relative_video_path)

        label_idx = item['label_idx'] 

        try:
            video_tensor = self._load_video_frames(image_path, clip_idx) 
            return video_tensor, label_idx
        except Exception as e:
            print(f"Error in __getitem__ for index {idx}, path {image_path}: {e}", file=sys.stderr)
            dummy = torch.zeros(3, self.frames, self.img_size, self.img_size) 
            return dummy, 0 


class R3DEmbedder(nn.Module):
    """R3D-18 network for video embeddings with optional SE blocks."""

    def __init__(self, pretrained=True, freeze_backbone=False, use_se=True):
        """Initialize the R3D embedder."""
        super(R3DEmbedder, self).__init__()

        if pretrained:
            weights = torchvision.models.video.R3D_18_Weights.DEFAULT
            self.backbone = torchvision.models.video.r3d_18(weights=weights)
        else:
            self.backbone = torchvision.models.video.r3d_18(weights=None)
        
        if use_se:
            self._add_se_blocks()

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        embedding_size = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity() 
        self.embedding_size = embedding_size 

    def _add_se_blocks(self):
        """Add Squeeze-and-Excitation blocks to the model."""
        print("Adding SE blocks to R3D-18 model")
        layer_channels = {'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512}
        for layer_name in ['layer3', 'layer4']:
            layer = getattr(self.backbone, layer_name)
            channels = layer_channels[layer_name]
            for i, block in enumerate(layer):
                se_block = nn.Sequential(
                    nn.AdaptiveAvgPool3d(1),
                    nn.Conv3d(channels, channels // 16, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(channels // 16, channels, kernel_size=1),
                    nn.Sigmoid()
                )
                setattr(block, 'se', se_block)
                def make_hook(blk):
                    def hook(module, input, output):
                        se_scale = blk.se(output)
                        return output * se_scale
                    return hook
                block.register_forward_hook(make_hook(block))

    def forward(self, x):
        """Forward pass through the R3D-18 network."""
        features = self.backbone(x)
        return features


class EmotionClassifier(nn.Module):
    """Classifier head for video features."""
    # Note: This class name is misleading now, it's a general video classifier.
    # It also assumes R3D embedder, needs adaptation for SlowFast.

    def __init__(self, num_classes=6, hidden_size=256, dropout=0.5, backbone='r3d_18', 
                 use_se=True, pretrained=True, freeze_backbone=False, embedding_size=512):
        """Initialize the classifier."""
        super(EmotionClassifier, self).__init__()
        
        print(f"WARNING: Using R3DEmbedder logic as placeholder for SlowFast backbone '{backbone}'. Needs proper SlowFast model loading.")
        self.video_embedder = R3DEmbedder( # Placeholder
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            use_se=use_se
        )
        self.embedding_size = self.video_embedder.embedding_size # Placeholder size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.embedding_size, num_classes)

    def forward(self, x):
        """Forward pass through the classifier."""
        if isinstance(x, list): 
             print("Warning: Model received list input, placeholder using first element.")
             features = self.video_embedder(x[0]) 
        else:
             features = self.video_embedder(x) 
        features = self.dropout(features)
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
        frames = frames.to(device) if isinstance(frames, torch.Tensor) else [f.to(device) for f in frames]
        labels = labels.to(device)
        optimizer.zero_grad()
        if use_amp:
            with autocast():
                outputs = model(frames)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        avg_loss = running_loss / (progress_bar.n + 1)
        acc = accuracy_score(all_labels, all_preds) * 100
        progress_bar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.1f}")
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds) * 100
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, test_clips=1):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    num_classes = model.classifier.out_features 
    target_names = [f"Class_{i}" for i in range(num_classes)]
    if num_classes == 2: 
        target_names = ['No Smile', 'Smile']
    elif num_classes == len(EMOTION_LABELS): 
         target_names = EMOTION_LABELS

    progress_bar = tqdm(dataloader, desc="Validation")
    with torch.no_grad():
        for batch_idx, (frames, labels) in enumerate(progress_bar):
            frames = frames.to(device) if isinstance(frames, torch.Tensor) else [f.to(device) for f in frames]
            labels = labels.to(device)
            outputs = model(frames)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            progress_bar.set_postfix(loss=f"{running_loss/(batch_idx+1):.4f}")
    val_loss = running_loss / len(dataloader)
    val_acc = accuracy_score(all_labels, all_preds) * 100
    conf_mat = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0)
    print("\nClassification Report:")
    print(class_report)
    return val_loss, val_acc


def save_checkpoint(model, optimizer, scheduler, epoch, val_acc, model_dir, model_name):
    """Save model checkpoint."""
    os.makedirs(model_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_acc': val_acc
    }
    checkpoint_path = os.path.join(model_dir, f"{model_name}_epoch{epoch:03d}.pt")
    torch.save(checkpoint, checkpoint_path)
    best_model_path = os.path.join(model_dir, f"{model_name}_best.pt")
    torch.save(checkpoint, best_model_path)
    return checkpoint_path, best_model_path


def load_yaml(file_path):
    """Load a YAML configuration file."""
    with open(file_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
            if '_BASE_' in config:
                base_config_path = os.path.join(os.path.dirname(file_path), config['_BASE_'])
                print(f"Loading base config: {base_config_path}")
                base_config = load_yaml(base_config_path)
                base_config.update(config)
                config = base_config
                del config['_BASE_'] 
        except yaml.YAMLError as exc:
            print(f"Error loading YAML file {file_path}: {exc}")
            sys.exit(1)
    return config


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train emotion recognition model with SlowFast backbone")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--manifest_file", type=str, required=True, help="Path to CSV manifest (train split)") 
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save models and logs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (overrides config)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (overrides config)")
    parser.add_argument("--img_size", type=int, default=None, help="Image size (overrides config)")
    parser.add_argument("--frames", type=int, default=None, help="Number of frames per clip (overrides config)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config)")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay (overrides config)")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout rate (overrides config)")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of dataloader workers (overrides config)")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze backbone weights (overrides config)")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training (overrides config)")
    args = parser.parse_args()
    
    config = load_yaml(args.config)
    
    def update_config(cfg_key_path, arg_value, cfg=config):
         if arg_value is not None:
            keys = cfg_key_path.split('.')
            d = cfg
            for key in keys[:-1]:
                d = d.setdefault(key, {})
            if arg_value is not None:
                 d[keys[-1]] = arg_value

    update_config('MODEL.DROPOUT_RATE', args.dropout)
    update_config('TRAIN.ENABLE', True) 
    update_config('TEST.ENABLE', True)  
    update_config('TRAIN.BATCH_SIZE', args.batch_size)
    update_config('TEST.BATCH_SIZE', args.batch_size) 
    update_config('SOLVER.MAX_EPOCH', args.epochs)
    update_config('SOLVER.BASE_LR', args.lr)
    update_config('SOLVER.WEIGHT_DECAY', args.weight_decay)
    update_config('DATA.NUM_FRAMES', args.frames)
    update_config('DATA.TRAIN_CROP_SIZE', args.img_size)
    update_config('DATA.TEST_CROP_SIZE', args.img_size) 
    update_config('DATA_LOADER.NUM_WORKERS', args.num_workers)
    if args.freeze_backbone: 
         update_config('MODEL.FREEZE_BACKBONE', True)
    if args.fp16: 
         update_config('TRAIN.MIXED_PRECISION', True)

    output_dir = args.output_dir
    config['OUTPUT_DIR'] = output_dir 
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_manifest_file = args.manifest_file
    if not os.path.exists(train_manifest_file):
        print(f"Error: Training manifest file not found: {train_manifest_file}", file=sys.stderr)
        sys.exit(1)
        
    val_manifest_file = train_manifest_file.replace("_train.csv", "_val.csv")
    if not os.path.exists(val_manifest_file):
         print(f"Warning: Validation manifest file not found at inferred path: {val_manifest_file}. Validation will fail.", file=sys.stderr)

    train_dataset = VideoDataset(
        manifest_file=train_manifest_file, 
        split="train",
        frames=config['DATA']['NUM_FRAMES'],
        img_size=config['DATA']['TRAIN_CROP_SIZE'],
        clips_per_video=config.get('TRAIN', {}).get('CLIPS_PER_VIDEO', 1), 
        config=config, 
        augment=True
    )

    val_dataset = VideoDataset(
        manifest_file=val_manifest_file, 
        split="val",
        frames=config['DATA']['NUM_FRAMES'],
        img_size=config['DATA']['TEST_CROP_SIZE'], 
        clips_per_video=1, 
        config=config, 
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['TRAIN']['BATCH_SIZE'], 
        shuffle=True,
        num_workers=config['DATA_LOADER']['NUM_WORKERS'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['TEST']['BATCH_SIZE'], 
        shuffle=False,
        num_workers=config['DATA_LOADER']['NUM_WORKERS'],
        pin_memory=True
    )
    
    print("--- Placeholder Model Creation ---")
    print("Loading R3D-18 based EmotionClassifier as a placeholder.")
    print("This needs to be replaced with actual SlowFast model loading and head adaptation.")
    
    model = EmotionClassifier( 
        num_classes=config['MODEL']['NUM_CLASSES'], 
        dropout=config['MODEL']['DROPOUT_RATE'],
    )
    
    model = model.to(device)
    
    optimizer_name = config['SOLVER']['OPTIMIZING_METHOD']
    if optimizer_name.lower() == 'adamw':
         optimizer = optim.AdamW(
            model.parameters(),
            lr=config['SOLVER']['BASE_LR'],
            weight_decay=config['SOLVER']['WEIGHT_DECAY']
        )
    elif optimizer_name.lower() == 'sgd':
         optimizer = optim.SGD(
            model.parameters(),
            lr=config['SOLVER']['BASE_LR'],
            momentum=config['SOLVER']['MOMENTUM'],
            weight_decay=config['SOLVER']['WEIGHT_DECAY']
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    lr_policy = config['SOLVER']['LR_POLICY']
    if lr_policy.lower() == 'onecycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config['SOLVER']['BASE_LR'],
            steps_per_epoch=len(train_loader),
            epochs=config['SOLVER']['MAX_EPOCH'],
        )
    elif lr_policy.lower() == 'cosine':
         scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config['SOLVER']['MAX_EPOCH'] * len(train_loader) 
        )
    else:
        print(f"Warning: LR policy '{lr_policy}' not explicitly handled, using CosineAnnealingLR as default.")
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config['SOLVER']['MAX_EPOCH'] * len(train_loader)
        )

    loss_func_name = config['MODEL']['LOSS_FUNC']
    if loss_func_name.lower() == 'cross_entropy':
        criterion = nn.CrossEntropyLoss() 
    else:
        raise ValueError(f"Unsupported loss function: {loss_func_name}")

    use_amp = config.get('TRAIN', {}).get('MIXED_PRECISION', False)
    scaler = GradScaler() if use_amp else None

    best_val_acc = 0.0
    patience_counter = 0 

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"slowfast_smile_{timestamp}" 

    val_accuracies = []
    train_accuracies = []
    val_losses = []
    train_losses = []

    config_path = os.path.join(output_dir, f"{model_name}_effective_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    print(f"Effective configuration saved to {config_path}")

    max_epochs = config['SOLVER']['MAX_EPOCH']
    for epoch in range(1, max_epochs + 1):
        print(f"\nEpoch {epoch}/{max_epochs}")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )

        val_loss, val_acc = validate(
            model, val_loader, criterion, device,
            test_clips=config.get('TEST', {}).get('NUM_ENSEMBLE_VIEWS', 1) 
        )

        train_accuracies.append(float(train_acc)) 
        train_losses.append(float(train_loss))
        val_accuracies.append(float(val_acc))
        val_losses.append(float(val_loss))

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            print(f"*** New best validation accuracy: {best_val_acc:.2f}% ***")
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_acc,
                output_dir, model_name 
            )

        if scheduler is not None and not isinstance(scheduler, OneCycleLR):
             scheduler.step()

    history = {
        'train_loss': train_losses,
        'train_acc': train_accuracies,
        'val_loss': val_losses,
        'val_acc': val_accuracies,
    }

    history_path = os.path.join(output_dir, f"{model_name}_history.json")
    try:
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"Training history saved to {history_path}")
    except Exception as e:
        print(f"Error saving training history: {e}", file=sys.stderr)

    print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved to {output_dir}/{model_name}_best.pt")


if __name__ == "__main__":
    try:
        import pandas
        import sklearn
        import yaml
    except ImportError as e:
        print(f"Missing required library: {e}. Please install requirements.", file=sys.stderr)
        sys.exit(1)

    main()
