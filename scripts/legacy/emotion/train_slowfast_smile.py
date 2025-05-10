#!/usr/bin/env python3
"""
Training script for SlowFast Smile Detection on CelebA.
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

# --- Helper Classes (Augmentations) ---
# (Keep augmentations as they might be useful, though less relevant for single images)
class RandomTimeReverse(nn.Module):
    def __init__(self, p=0.5): super().__init__(); self.p = p
    def forward(self, x): return torch.flip(x, dims=[0]) if random.random() < self.p else x

class RandomDropFrames(nn.Module):
    def __init__(self, max_drop=2, p=0.5): super().__init__(); self.max_drop = max_drop; self.p = p
    def forward(self, x):
        if random.random() < self.p:
            T = x.shape[0]; drop_count = random.randint(1, min(self.max_drop, T-1))
            drop_indices = sorted(random.sample(range(T), drop_count)); keep_indices = list(range(T))
            for idx in drop_indices: keep_indices.remove(idx)
            while len(keep_indices) < T: keep_indices.append(keep_indices[-1])
            return x[keep_indices]
        return x

class Cutout(nn.Module):
    def __init__(self, size=10, count=2, p=0.5): super().__init__(); self.size = size; self.count = count; self.p = p
    def forward(self, x):
        if random.random() < self.p:
            T, C, H, W = x.shape; mask = torch.ones_like(x)
            for _ in range(self.count):
                y = random.randint(0, H - self.size); x_pos = random.randint(0, W - self.size)
                mask[:, :, y:y+self.size, x_pos:x_pos+self.size] = 0
            return x * mask
        return x

# --- Dataset Class (Adapted for CelebA Smile) ---
class SmileDataset(Dataset):
    """Dataset loader for CelebA smile detection."""

    def __init__(self, manifest_file, split='train', config=None):
        self.manifest_file = manifest_file
        self.split = split
        self.config = config or {}
        
        # Extract relevant config sections
        data_cfg = self.config.get('DATA', {})
        aug_cfg = self.config.get('AUGMENTATION', {}) # Assuming augmentations are under this key in base config
        
        self.frames = data_cfg.get('NUM_FRAMES', 64) # Default if not specified
        self.img_size = data_cfg.get('TRAIN_CROP_SIZE', 224) if split == 'train' else data_cfg.get('TEST_CROP_SIZE', 256)
        self.augment = split == 'train' and self.config.get('TRAIN', {}).get('ENABLE', False) # Check if training is enabled

        # Load the manifest
        try:
            df = pd.read_csv(manifest_file)
        except FileNotFoundError:
             print(f"Error: Manifest file not found: {manifest_file}", file=sys.stderr)
             sys.exit(1)

        if 'smile_label' not in df.columns:
             raise ValueError(f"Manifest file {manifest_file} must contain a 'smile_label' column.")
        if 'rel_video' not in df.columns: # Check for image path column
             raise ValueError(f"Manifest file {manifest_file} must contain a 'rel_video' (relative image path) column.")
             
        df['label_idx'] = df['smile_label'] # Use the smile_label column directly
        self.data = df

        print(f"Smile label distribution for {split} split ({os.path.basename(manifest_file)}):")
        print(df['smile_label'].value_counts())
        
        # Setup transforms
        self.transform = self._get_transforms(aug_cfg)

    def _get_transforms(self, aug_cfg):
        """Create augmentation pipeline based on config."""
        transform_list = []
        if self.augment:
            if aug_cfg.get('RANDOM_FLIP', True): # Use keys potentially from base config
                transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
            # Add other augmentations based on config keys (e.g., COLOR_JITTER, RAND_AUGMENT)
            # Example:
            # if aug_cfg.get('COLOR_JITTER', {}).get('ENABLE', False):
            #     transform_list.append(transforms.ColorJitter(...))
        
        # Note: ToTensor and Normalize are applied in _load_image_frames
        return transforms.Compose(transform_list)

    def __len__(self):
        return len(self.data)

    def _load_image_frames(self, image_path):
        """Load single image and repeat it to match expected frame count."""
        try:
            frame = cv2.imread(image_path)
            if frame is None: raise IOError(f"Could not read image file: {image_path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.img_size, self.img_size))
            pil_img = Image.fromarray(frame)
            augmented_pil_img = self.transform(pil_img) 
            frame_tensor = TF.to_tensor(augmented_pil_img)
            normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            frame_tensor = normalizer(frame_tensor)
            # Repeat frame: [C, H, W] -> [C, T, H, W]
            video_tensor = frame_tensor.unsqueeze(1).repeat(1, self.frames, 1, 1) 
            return video_tensor
        except Exception as e:
             print(f"Error processing image {image_path}: {e}", file=sys.stderr)
             return torch.zeros(3, self.frames, self.img_size, self.img_size) # [C, T, H, W]

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        base_path = self.config.get('DATA', {}).get('PATH_PREFIX', '/home/ubuntu/datasets/') 
        relative_image_path = item['rel_video'] 
        image_path = os.path.join(base_path, relative_image_path)
        label_idx = item['label_idx'] 
        video_tensor = self._load_image_frames(image_path) 
        # SlowFast expects input as a list: [slow_path_input, fast_path_input]
        # For image classification, we can potentially feed the same tensor to both
        # or use a model adapted for single pathway input. Assuming list input for now.
        # Alpha parameter controls fast pathway sampling rate, get from config
        alpha = self.config.get('SLOWFAST', {}).get('ALPHA', 4) # Default alpha if not in config
        fast_pathway = video_tensor
        # Sample frames for slow pathway
        slow_idx = torch.linspace(0, self.frames - 1, self.frames // alpha).long()
        slow_pathway = video_tensor[:, slow_idx, :, :]
        
        return [slow_pathway, fast_pathway], label_idx

# --- Training and Validation Functions ---
def train_epoch(model, dataloader, criterion, optimizer, scaler, device, use_amp=True):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    progress_bar = tqdm(dataloader, desc="Training")
    for inputs, labels in progress_bar:
        # Input might be a list for SlowFast
        if isinstance(inputs, list):
            inputs = [inp.to(device) for inp in inputs]
        else:
            inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        if use_amp:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
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

def validate(model, dataloader, criterion, device, config):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    num_classes = config.get('MODEL', {}).get('NUM_CLASSES', 2)
    target_names = ['No Smile', 'Smile'] if num_classes == 2 else [f"Class_{i}" for i in range(num_classes)]

    progress_bar = tqdm(dataloader, desc="Validation")
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            if isinstance(inputs, list):
                inputs = [inp.to(device) for inp in inputs]
            else:
                inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            progress_bar.set_postfix(loss=f"{running_loss/(batch_idx+1):.4f}")
            
    val_loss = running_loss / len(dataloader)
    val_acc = accuracy_score(all_labels, all_preds) * 100
    print("\nValidation Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))
    return val_loss, val_acc

# --- Checkpoint Function ---
def save_checkpoint(model, optimizer, scheduler, epoch, val_acc, model_dir, model_name):
    os.makedirs(model_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch, 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_acc': val_acc
    }
    best_model_path = os.path.join(model_dir, f"{model_name}_best.pt")
    torch.save(checkpoint, best_model_path)
    print(f"Checkpoint saved to {best_model_path}")

# --- YAML Loading ---
def load_yaml(file_path):
    """Load YAML, handling _BASE_ includes."""
    with open(file_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
            if '_BASE_' in config:
                base_path = os.path.join(os.path.dirname(file_path), config['_BASE_'])
                if os.path.exists(base_path):
                    print(f"Loading base config: {base_path}")
                    base_config = load_yaml(base_path) # Recursive call
                    # Deep merge config into base_config
                    # (Simple update might not merge nested dicts correctly)
                    def deep_update(d, u):
                        for k, v in u.items():
                            if isinstance(v, dict):
                                d[k] = deep_update(d.get(k, {}), v)
                            else:
                                d[k] = v
                        return d
                    config = deep_update(base_config, config)
                    del config['_BASE_']
                else:
                     print(f"Warning: Base config file not found: {base_path}", file=sys.stderr)
        except yaml.YAMLError as exc:
            print(f"Error loading YAML file {file_path}: {exc}", file=sys.stderr)
            sys.exit(1)
    return config

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Train SlowFast Smile Detection")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    # Remove other args, rely solely on config file
    args = parser.parse_args()

    # Load configuration
    config = load_yaml(args.config)
    print("--- Loaded Configuration ---")
    print(yaml.dump(config))
    print("--------------------------")

    output_dir = config.get('OUTPUT_DIR', 'checkpoints/smile_slowfast_default')
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Dataset and Dataloader ---
    data_cfg = config.get('DATA', {})
    train_manifest = os.path.join(data_cfg.get('PATH_TO_DATA_DIR', 'datasets'), "celeba_smile_train.csv")
    val_manifest = os.path.join(data_cfg.get('PATH_TO_DATA_DIR', 'datasets'), "celeba_smile_val.csv")

    if not os.path.exists(train_manifest):
        print(f"Error: Training manifest not found: {train_manifest}", file=sys.stderr); sys.exit(1)
    if not os.path.exists(val_manifest):
        print(f"Error: Validation manifest not found: {val_manifest}", file=sys.stderr); sys.exit(1)

    train_dataset = SmileDataset(manifest_file=train_manifest, split="train", config=config)
    val_dataset = SmileDataset(manifest_file=val_manifest, split="val", config=config)

    train_loader = DataLoader(
        train_dataset, batch_size=config.get('TRAIN', {}).get('BATCH_SIZE', 16), shuffle=True,
        num_workers=config.get('DATA_LOADER', {}).get('NUM_WORKERS', 4), pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.get('TEST', {}).get('BATCH_SIZE', 16), shuffle=False,
        num_workers=config.get('DATA_LOADER', {}).get('NUM_WORKERS', 4), pin_memory=True
    )

    # --- Model ---
    # Load the actual SlowFast model using the configuration
    # Ensure slowfast library is available and import necessary components
    # Note: Adjust path if slowfast is not installed globally but part of the project
    # sys.path.append('/path/to/PySlowFast') # Example if needed
    from slowfast.models import build_model
    from fvcore.common.config import CfgNode # Assuming config is dict, convert if needed

    # Convert the loaded dictionary config to CfgNode if required by build_model
    # This assumes the YAML structure matches what CfgNode expects
    cfg_for_model = CfgNode(config)

    try: # Add a try block specifically for the build_model call
        model = build_model(cfg_for_model)
        print("--- Successfully loaded SlowFast model ---")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to build SlowFast model: {e}", file=sys.stderr)
        print("Check config structure and PySlowFast compatibility.", file=sys.stderr)
        sys.exit(1) # Exit on build failure

    model = model.to(device)

    # --- Optimizer ---
    solver_cfg = config.get('SOLVER', {})
    optimizer_name = solver_cfg.get('OPTIMIZING_METHOD', 'sgd')
    if optimizer_name.lower() == 'adamw':
         optimizer = optim.AdamW(model.parameters(), lr=solver_cfg.get('BASE_LR', 0.001), weight_decay=solver_cfg.get('WEIGHT_DECAY', 1e-4))
    elif optimizer_name.lower() == 'sgd':
         # Ensure all retrieved config values are cast to float
         optimizer = optim.SGD(
             model.parameters(),
             lr=float(solver_cfg.get('BASE_LR', 0.1)), 
             momentum=float(solver_cfg.get('MOMENTUM', 0.9)), 
             weight_decay=float(solver_cfg.get('WEIGHT_DECAY', 1e-4)) # Cast the retrieved value
         )
    else: raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # --- Scheduler ---
    lr_policy = solver_cfg.get('LR_POLICY', 'cosine')
    max_epochs = solver_cfg.get('MAX_EPOCH', 20)
    if lr_policy.lower() == 'onecycle':
        scheduler = OneCycleLR(optimizer, max_lr=solver_cfg.get('BASE_LR', 0.001), steps_per_epoch=len(train_loader), epochs=max_epochs)
    elif lr_policy.lower() == 'cosine':
         scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs * len(train_loader))
    else:
        print(f"Warning: LR policy '{lr_policy}' not explicitly handled, using CosineAnnealingLR.")
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs * len(train_loader))

    # --- Loss Function ---
    loss_func_name = config.get('MODEL', {}).get('LOSS_FUNC', 'cross_entropy')
    if loss_func_name.lower() == 'cross_entropy':
        criterion = nn.CrossEntropyLoss() 
    else: raise ValueError(f"Unsupported loss function: {loss_func_name}")

    # --- Training Loop ---
    use_amp = config.get('TRAIN', {}).get('MIXED_PRECISION', False)
    scaler = GradScaler() if use_amp else None
    best_val_acc = 0.0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"slowfast_smile_{timestamp}"
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    config_path = os.path.join(output_dir, f"{model_name}_effective_config.yaml")
    with open(config_path, 'w') as f: yaml.dump(config, f)
    print(f"Effective configuration saved to {config_path}")

    for epoch in range(1, max_epochs + 1):
        print(f"\nEpoch {epoch}/{max_epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
        val_loss, val_acc = validate(model, val_loader, criterion, device, config)
        
        history['train_loss'].append(float(train_loss)); history['train_acc'].append(float(train_acc))
        history['val_loss'].append(float(val_loss)); history['val_acc'].append(float(val_acc))

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            print(f"*** New best validation accuracy: {best_val_acc:.2f}% ***")
            save_checkpoint(model, optimizer, scheduler, epoch, val_acc, output_dir, model_name)

        if scheduler is not None and not isinstance(scheduler, OneCycleLR):
             scheduler.step()

    # --- Save History ---
    history_path = os.path.join(output_dir, f"{model_name}_history.json")
    try:
        with open(history_path, 'w') as f: json.dump(history, f, indent=4)
        print(f"Training history saved to {history_path}")
    except Exception as e: print(f"Error saving training history: {e}", file=sys.stderr)

    print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved to {output_dir}/{model_name}_best.pt")

if __name__ == "__main__":
    try:
        import pandas; import sklearn; import yaml 
    except ImportError as e:
        print(f"Missing required library: {e}. Please install requirements.", file=sys.stderr)
        sys.exit(1)
    main()
