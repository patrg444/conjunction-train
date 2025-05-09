#!/usr/bin/env python3
"""
Multimodal Humor Detection Training Script (Fusion Architecture)
"""

import os
import sys
# Add project root to path to allow sibling imports (ser_hubert, dataloaders)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added project root to sys.path: {project_root}")

import argparse
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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import torchvision
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import torchaudio # For audio loading
from transformers import AutoFeatureExtractor, AutoModel # Added AutoModel

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from datetime import datetime
import logging # Added for better logging

# Import the correct classifier from the script that generated the checkpoint
# Assuming the R3D-18 based classifier is defined here or accessible
try:
    # Use relative import since both files are in the 'scripts' directory
    from .train_slowfast_emotion import EmotionClassifier as VideoEmotionClassifier
    print("Successfully imported EmotionClassifier from .train_slowfast_emotion")
except ImportError as e:
    print(f"Warning: Could not import EmotionClassifier from .train_slowfast_emotion: {e}. Video branch might fail if used.")
    VideoEmotionClassifier = None # Set to None if import fails

# Import the Hubert SER model
try:
    # Assuming ser_hubert is adjacent or in python path
    from ser_hubert.hubert_ser_module import HubertSER
    print("Successfully imported HubertSER from ser_hubert.hubert_ser_module")
except ImportError as e:
    print(f"Error importing HubertSER: {e}")
    print("Ensure ser_hubert directory is accessible.")
    sys.exit(1)

# Import the HumorDataset
try:
    from dataloaders.humor_dataset import HumorDataset
    print("Successfully imported HumorDataset from dataloaders.humor_dataset")
except ImportError as e:
    print(f"Error importing HumorDataset: {e}")
    print("Ensure dataloaders/humor_dataset.py exists and the project structure is correct.")
    sys.exit(1)


# Labels for Humor Task
HUMOR_LABELS = ['Non-Laugh', 'Laugh']

# --- Custom Collate Function ---
def safe_collate_fn(batch):
    """
    Collate function that filters out None samples returned by the dataset.
    """
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        # Return None or an empty dict if the batch becomes empty after filtering
        # Returning None might require handling in the training loop
        # Returning an empty dict might also cause issues depending on loop structure
        # Let's return None for now and adjust the loop if needed.
        # Alternatively, could raise an error or return a dummy batch, but skipping seems best.
        logging.warning("Batch Collapsed: All samples in this batch failed to load.")
        return None # Signal to skip this batch
    # If batch is not empty, use default collate
    return torch.utils.data.dataloader.default_collate(batch)


# --- Augmentation Classes (Copied from train_slowfast_emotion.py - Keep for video branch) ---
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

# --- Helper Functions ---
def load_yaml(file_path):
    """Load a YAML configuration file."""
    try:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading YAML file {file_path}: {e}")
        return None # Return None or raise error

# --- Dataset Definition (REMOVED - Now importing HumorDataset) ---
# class FusionDataset(Dataset): ... (Removed)

# --- Model Definition ---

class FusionModel(nn.Module):
    # Updated to accept text_model_name
    def __init__(self, num_classes_humor, video_checkpoint_path, hubert_checkpoint_path, hubert_model_name="facebook/hubert-base-ls960", text_model_name="distilbert-base-uncased", fusion_dim=512, dropout=0.5, use_video=True, use_text=True): # Added text_model_name, use_text
        super().__init__()
        self.num_classes_humor = num_classes_humor
        self.use_video = use_video
        self.use_text = use_text # Flag to control text branch usage

        # --- Video Branch (Conditional) ---
        self.video_embedder = None
        self.video_embedding_size = 0 # Default to 0 if no video
        if self.use_video:
            if VideoEmotionClassifier is None:
                 raise RuntimeError("Video branch requested but VideoEmotionClassifier failed to import.")
            print("Instantiating VideoEmotionClassifier (R3D-18 based)...")
            # Assuming the imported classifier has the necessary structure
            self.video_embedder = VideoEmotionClassifier(
                num_classes=6, # Original num_classes for loading checkpoint (e.g., 6 emotions)
                use_se=True, # Assuming SE blocks were used in pre-trained model
                pretrained=False # We load weights manually
            )
            self.video_embedding_size = self.video_embedder.embedding_size # Should be 512

            if video_checkpoint_path and os.path.exists(video_checkpoint_path):
                print(f"Loading Video (R3D-18) weights from checkpoint: {video_checkpoint_path}")
                video_checkpoint = torch.load(video_checkpoint_path, map_location='cpu')
                video_state_dict = video_checkpoint.get('model_state_dict', video_checkpoint)
                # Load weights, potentially ignoring the final classifier if sizes mismatch
                load_result_vid = self.video_embedder.load_state_dict(video_state_dict, strict=False)
                print(f"Loaded Video Embedder state_dict. Load result (strict=False): {load_result_vid}")
                if load_result_vid.missing_keys or load_result_vid.unexpected_keys:
                     print("  Note: Mismatched keys likely due to loading a pre-trained model or different final layer.")
                     print(f"  Missing keys: {load_result_vid.missing_keys}")
                     print(f"  Unexpected keys: {load_result_vid.unexpected_keys}")
            else:
                 print(f"Warning: Video checkpoint not found or not provided ({video_checkpoint_path}). Video branch will use random weights.")

            self.video_embedder.classifier = nn.Identity() # Remove final layer
            print("Freezing Video (R3D-18) backbone parameters.")
            # Freeze only the backbone, not the whole embedder if fine-tuning later
            if hasattr(self.video_embedder, 'video_embedder'):
                 for param in self.video_embedder.video_embedder.parameters():
                     param.requires_grad = False
            else:
                 print("Warning: Could not find 'video_embedder' attribute to freeze backbone.")

        # --- Audio Branch (Hubert SER) ---
        print(f"Instantiating HubertSER model ({hubert_model_name})...")
        # Instantiate HubertSER - num_classes might differ, handle during loading
        self.hubert_ser = HubertSER(
            hubert_name=hubert_model_name,
            num_classes=6 # Use original num_classes for loading checkpoint (e.g., 6 emotions)
        )
        if hubert_checkpoint_path and os.path.exists(hubert_checkpoint_path):
            print(f"Loading Hubert SER weights from checkpoint: {hubert_checkpoint_path}")
            hub_checkpoint = torch.load(hubert_checkpoint_path, map_location='cpu')
            # Check if it's a PL checkpoint with 'state_dict' key
            if 'state_dict' in hub_checkpoint:
                hub_state_dict = hub_checkpoint['state_dict']
                # Adjust keys if needed (e.g., remove 'model.' prefix if saved by PL)
                hub_state_dict = {k.replace("model.", ""): v for k, v in hub_state_dict.items()}
            else:
                hub_state_dict = hub_checkpoint # Assume it's just the state dict

            # Load weights, potentially ignoring the final classifier if sizes mismatch
            load_result_hub = self.hubert_ser.load_state_dict(hub_state_dict, strict=False)
            print(f"Loaded Hubert SER state_dict. Load result (strict=False): {load_result_hub}")
            if load_result_hub.missing_keys or load_result_hub.unexpected_keys:
                print("  Note: Mismatched keys likely due to loading a pre-trained model or different final layer.")
                print(f"  Missing keys: {load_result_hub.missing_keys}")
                print(f"  Unexpected keys: {load_result_hub.unexpected_keys}")
        else:
            print(f"Warning: Hubert checkpoint not found or not provided ({hubert_checkpoint_path}). Hubert branch will use random weights.")


        # Get feature dimension *before* the final FC layer of HubertSER
        self.hubert_feature_dim = self.hubert_ser.fc.in_features # Get dim before replacing
        self.hubert_ser.fc = nn.Identity() # Remove final layer

        # Freeze Hubert backbone (optional, default True for feature extraction)
        print("Freezing Hubert backbone parameters.")
        for param in self.hubert_ser.hubert.parameters():
            param.requires_grad = False
        # Keep dropout layer trainable? Or freeze it too? Let's freeze for now.
        # for param in self.hubert_ser.dropout.parameters():
        #     param.requires_grad = False

        # --- Text Branch (Conditional) ---
        self.text_embedder = None
        self.text_embedding_size = 0
        if self.use_text:
            print(f"Instantiating Text Embedder model ({text_model_name})...")
            try:
                self.text_embedder = AutoModel.from_pretrained(text_model_name)
                # Determine embedding size (e.g., from config or model's hidden_size)
                # For models like DistilBERT, it's often in config.hidden_size
                self.text_embedding_size = self.text_embedder.config.hidden_size
                print(f"Text embedding size: {self.text_embedding_size}")
                # Freeze text backbone (optional, default True)
                print("Freezing Text Embedder backbone parameters.")
                for param in self.text_embedder.parameters():
                    param.requires_grad = False
            except Exception as e:
                print(f"Error loading text model {text_model_name}: {e}. Text branch disabled.")
                self.use_text = False # Disable text if model loading fails
                self.text_embedder = None
                self.text_embedding_size = 0


        # --- Fusion Layer ---
        # Use the determined feature dimensions
        print(f"Video embedding size: {self.video_embedding_size}, Hubert feature dim: {self.hubert_feature_dim}, Text embedding size: {self.text_embedding_size}")
        fusion_input_dim = self.hubert_feature_dim + \
                           (self.video_embedding_size if self.use_video else 0) + \
                           (self.text_embedding_size if self.use_text else 0)
        print(f"Fusion input dimension: {fusion_input_dim}")
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Final Classifier for Humor (2 classes)
        self.classifier = nn.Linear(fusion_dim, self.num_classes_humor)

    def _pool_text(self, last_hidden_state, attention_mask):
        """Mean pool the last hidden state using the attention mask."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, audio_input_values, audio_attention_mask, video_input=None, text_input_ids=None, text_attention_mask=None):
        # audio_input_values shape: [B, AudioSeqLen]
        # audio_attention_mask shape: [B, AudioSeqLen]
        # video_input shape: [B, C, T, H, W] (optional, already permuted)
        # text_input_ids shape: [B, TextSeqLen] (optional)
        # text_attention_mask shape: [B, TextSeqLen] (optional)

        # Get Hubert features (before final FC)
        hubert_outputs = self.hubert_ser.hubert(input_values=audio_input_values, attention_mask=audio_attention_mask)
        # Use the pooling method defined in HubertSER if available, otherwise default pool
        if hasattr(self.hubert_ser, '_pool'):
             hubert_features = self.hubert_ser._pool(hubert_outputs.last_hidden_state, audio_attention_mask)
        else: # Fallback pooling (e.g., mean pool) - adjust if needed
             hubert_features = torch.mean(hubert_outputs.last_hidden_state, dim=1)
        # hubert_features shape: [B, hubert_feature_dim]

        # --- Collect features from active modalities ---
        active_features = [hubert_features]

        if self.use_video and video_input is not None and self.video_embedder is not None:
            video_features = self.video_embedder(video_input) # Shape: [B, video_embedding_size]
            active_features.append(video_features)

        if self.use_text and text_input_ids is not None and text_attention_mask is not None and self.text_embedder is not None:
            text_outputs = self.text_embedder(input_ids=text_input_ids, attention_mask=text_attention_mask)
            # Pool the text embeddings (e.g., mean pooling of last hidden state)
            text_features = self._pool_text(text_outputs.last_hidden_state, text_attention_mask) # Shape: [B, text_embedding_size]
            active_features.append(text_features)

        # Concatenate features from active modalities
        if len(active_features) > 1:
            fused_features = torch.cat(active_features, dim=1)
        else:
            fused_features = active_features[0] # Only hubert features if others are disabled/failed

        # Pass through fusion layer
        # Ensure the fusion layer's input dim matches the concatenated feature size
        fused_output = self.fusion_layer(fused_features)

        # Classify for humor
        logits = self.classifier(fused_output)

        return logits

# --- Training and Validation Functions ---
# Updated to handle HumorDataset output format including text
def train_epoch(model, dataloader, criterion, optimizer, scaler, device, use_amp, use_video, use_text): # Added use_text
    model.train()
    # Ensure frozen backbones remain in eval mode
    if hasattr(model, 'hubert_ser') and not any(p.requires_grad for p in model.hubert_ser.hubert.parameters()):
        model.hubert_ser.hubert.eval()
    # Ensure Video backbone remains in eval mode if frozen and used
    if use_video and hasattr(model, 'video_embedder') and model.video_embedder and hasattr(model.video_embedder, 'video_embedder') and not any(p.requires_grad for p in model.video_embedder.video_embedder.parameters()):
        model.video_embedder.video_embedder.eval()
    # Ensure Text backbone remains in eval mode if frozen and used
    if use_text and hasattr(model, 'text_embedder') and model.text_embedder and not any(p.requires_grad for p in model.text_embedder.parameters()):
        model.text_embedder.eval()


    running_loss = 0.0
    all_preds = []
    all_labels = []
    progress_bar = tqdm(dataloader, desc="Training Humor")

    # Adjusted loop to unpack HumorDataset sample dictionary
    # Add check for None batch returned by safe_collate_fn
    for batch in progress_bar:
        if batch is None: # Skip batch if collate_fn returned None
            continue
        audio_input_values = batch['audio_input_values'].to(device)
        audio_attention_mask = batch['audio_attention_mask'].to(device)
        labels = batch['laugh_label'].to(device) # Use laugh_label

        video_data = None
        if use_video and 'video' in batch:
            video_data = batch['video'].to(device)
            # Ensure video data has the correct shape [B, T, C, H, W] -> [B, C, T, H, W] for R3D
            if video_data.dim() == 5 and video_data.shape[2] == 3: # If shape is B, T, C, H, W
                 video_data = video_data.permute(0, 2, 1, 3, 4) # Permute to B, C, T, H, W

        text_input_ids = None
        text_attention_mask = None
        if use_text and 'text_input_ids' in batch and 'text_attention_mask' in batch:
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            # Pass data to the model, including text if used
            outputs = model(audio_input_values=audio_input_values,
                            audio_attention_mask=audio_attention_mask,
                            video_input=video_data,
                            text_input_ids=text_input_ids,
                            text_attention_mask=text_attention_mask)
            loss = criterion(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            # Optional: Gradient clipping
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        progress_bar.set_postfix(loss=loss.item())


    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds) * 100
    epoch_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    return epoch_loss, epoch_acc, epoch_f1

def validate(model, dataloader, criterion, device, use_video, use_text): # Added use_text
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    progress_bar = tqdm(dataloader, desc="Validating Humor")

    with torch.no_grad():
        # Adjusted loop to unpack HumorDataset sample dictionary
        # Add check for None batch returned by safe_collate_fn
        for batch in progress_bar:
            if batch is None: # Skip batch if collate_fn returned None
                continue
            audio_input_values = batch['audio_input_values'].to(device)
            audio_attention_mask = batch['audio_attention_mask'].to(device)
            labels = batch['laugh_label'].to(device) # Use laugh_label

            video_data = None
            if use_video and 'video' in batch:
                video_data = batch['video'].to(device)
                # Ensure video data has the correct shape [B, T, C, H, W] -> [B, C, T, H, W] for R3D
                if video_data.dim() == 5 and video_data.shape[2] == 3: # If shape is B, T, C, H, W
                    video_data = video_data.permute(0, 2, 1, 3, 4) # Permute to B, C, T, H, W

            text_input_ids = None
            text_attention_mask = None
            if use_text and 'text_input_ids' in batch and 'text_attention_mask' in batch:
                text_input_ids = batch['text_input_ids'].to(device)
                text_attention_mask = batch['text_attention_mask'].to(device)

            # Pass data to the model
            outputs = model(audio_input_values=audio_input_values,
                            audio_attention_mask=audio_attention_mask,
                            video_input=video_data,
                            text_input_ids=text_input_ids,
                            text_attention_mask=text_attention_mask)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / len(dataloader)
    val_acc = accuracy_score(all_labels, all_preds) * 100
    val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100 # Added F1 score
    print("\nValidation Classification Report (Humor):")
    # Use HUMOR_LABELS for report
    print(classification_report(all_labels, all_preds, target_names=HUMOR_LABELS, zero_division=0))
    return val_loss, val_acc, val_f1

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Train Multimodal Humor Detection Model")
    # Config path is primary
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file (e.g., configs/train_humor.yaml)")

    # Allow overrides via command line (optional)
    parser.add_argument("--output_dir", type=str, help="Override output directory from config")
    parser.add_argument("--epochs", type=int, help="Override number of epochs from config")
    parser.add_argument("--batch_size", type=int, help="Override batch size from config")
    parser.add_argument("--lr", type=float, help="Override learning rate from config")
    parser.add_argument("--use_video", action=argparse.BooleanOptionalAction, help="Override use_video setting from config (use --use_video or --no-use_video)")
    parser.add_argument("--use_text", action=argparse.BooleanOptionalAction, default=True, help="Override use_text setting from config (use --use_text or --no-use_text, default: True)") # Added use_text override
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True, help="Use mixed precision training (--fp16 or --no-fp16, default: True)")
    parser.add_argument("--early_stop", type=int, default=10, help="Early stopping patience") # Keep as arg for convenience

    args = parser.parse_args()

    # --- Load Configuration ---
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)
    config = load_yaml(args.config)
    if config is None:
        sys.exit(1)

    # --- Apply Command Line Overrides ---
    output_dir = args.output_dir or config['checkpointing']['checkpoint_dir'] # Use checkpoint_dir as base output
    log_dir = args.output_dir or config['logging']['log_dir'] # Use log_dir as base output
    epochs = args.epochs or config['training']['epochs']
    batch_size = args.batch_size or config['data']['dataloader_params']['batch_size']
    lr = args.lr or config['training']['optimizer_params']['lr']
    weight_decay = config['training']['optimizer_params'].get('weight_decay', 1e-5) # Get from config or default
    label_smoothing = config['training'].get('label_smoothing', 0.1) # Get from config or default
    num_workers = config['data']['dataloader_params'].get('num_workers', 4)
    pin_memory = config['data']['dataloader_params'].get('pin_memory', True)
    use_amp = args.fp16 # Use command line arg for fp16 override

    # Determine if video should be used (command line overrides config)
    if args.use_video is not None:
        use_video = args.use_video
    else:
        # Infer from model config if possible, default to False if ambiguous
        # This assumes the config might have a direct 'use_video' flag or implies it by architecture
        use_video = config['model'].get('use_video', False) # Default to False if not specified

    # Determine if text should be used (command line overrides config)
    if args.use_text is not None:
        use_text = args.use_text
    else:
        use_text = config['model'].get('use_text', True) # Default to True if not specified

    # --- Argument Validation (Post-Config) ---
    video_checkpoint = config['model'].get('video_checkpoint', None) # Get from config
    hubert_checkpoint = config['model'].get('hubert_checkpoint', None) # Get from config
    hubert_model_name = config['model'].get('hubert_model_name', "facebook/hubert-base-ls960") # Get from config
    text_model_name = config['model'].get('text_model_name', "distilbert-base-uncased") # Get from config

    if use_video and not video_checkpoint:
        print("Warning: --use_video is set but no 'video_checkpoint' found in config. Video branch will use random weights.")
    if not hubert_checkpoint:
        print("Warning: No 'hubert_checkpoint' found in config. Hubert branch will use random weights.")
    if use_text and not text_model_name:
         print("Warning: --use_text is set but no 'text_model_name' found in config. Text branch cannot be initialized.")
         use_text = False # Disable text if model name is missing
    # --- End Argument Validation ---

    # Setup device, output dir, etc.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True) # Ensure log dir exists
    print(f"Using device: {device}")
    print(f"Using Video: {use_video}")
    print(f"Using Text: {use_text}") # Added log for text usage
    print(f"Using Mixed Precision (AMP): {use_amp}")

    # Create Datasets and Dataloaders using HumorDataset
    print("Creating Humor Datasets...")
    dataset_params = config['data'].get('dataset_params', {})
    # Look for manifests with text first, fallback to original names
    train_csv_path = config['data'].get('train_csv_with_text', config['data'].get('train_csv'))
    val_csv_path = config['data'].get('val_csv_with_text', config['data'].get('val_csv'))
    text_model_name = config['model'].get('text_model_name', "distilbert-base-uncased") # Get text model name
    max_text_len = dataset_params.get('max_text_len', 128) # Get max text length

    if not train_csv_path:
        print("Error: 'train_csv' or 'train_csv_with_text' path not found in config data section.")
        sys.exit(1)

    train_dataset = HumorDataset(
        manifest_path=train_csv_path, # Use train_csv
        dataset_root=config['data']['dataset_root'],
        duration=dataset_params.get('duration', 1.0),
        sample_rate=dataset_params.get('sample_rate', 16000),
        video_fps=dataset_params.get('video_fps', 15),
        img_size=dataset_params.get('img_size', 112),
        hubert_model_name=hubert_model_name,
        text_model_name=text_model_name, # Pass text model name
        max_text_len=max_text_len,       # Pass max text length
        split="train", # Keep split logic if HumorDataset uses it internally
        augment=True
    )

    val_dataset = None
    if val_csv_path:
        print(f"Validation manifest found: {val_csv_path}")
        val_dataset = HumorDataset(
            manifest_path=val_csv_path, # Use val_csv
            dataset_root=config['data']['dataset_root'],
            duration=dataset_params.get('duration', 1.0),
            sample_rate=dataset_params.get('sample_rate', 16000),
            video_fps=dataset_params.get('video_fps', 15),
        img_size=dataset_params.get('img_size', 112),
        hubert_model_name=hubert_model_name,
        text_model_name=text_model_name, # Pass text model name
        max_text_len=max_text_len,       # Pass max text length
        split="val", # Keep split logic if HumorDataset uses it internally
        augment=False # No augmentation for validation
    )
    else:
        print("No 'val_csv' path found in config. Validation will be skipped.")


    # Use the safe_collate_fn to handle None samples
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, collate_fn=safe_collate_fn)
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=safe_collate_fn)

    print("Datasets created.")

    # Create Model
    print("Creating Fusion Model for Humor Detection...")
    model_params = config['model']
    model = FusionModel(
        num_classes_humor=model_params.get('num_classes_humor', 2), # Default to 2
        video_checkpoint_path=video_checkpoint,
        hubert_checkpoint_path=hubert_checkpoint,
        hubert_model_name=hubert_model_name,
        text_model_name=text_model_name, # Pass text model name
        fusion_dim=model_params.get('fusion_dim', 512),
        dropout=model_params.get('dropout', 0.5),
        use_video=use_video, # Pass the flag
        use_text=use_text    # Pass the flag
    ).to(device)
    print("Model created.")

    # Optimizer, Scheduler, Loss
    optimizer_name = config['training'].get('optimizer', 'AdamW')
    if optimizer_name.lower() == 'adamw':
         optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adam':
         optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
         print(f"Warning: Unsupported optimizer '{optimizer_name}'. Defaulting to AdamW.")
         optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs) # Example
    scheduler = None # Keep it simple for now
    loss_name = config['training'].get('loss_fn', 'CrossEntropyLoss')
    if loss_name == 'CrossEntropyLoss':
         criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
         print(f"Warning: Unsupported loss function '{loss_name}'. Defaulting to CrossEntropyLoss.")
         criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    scaler = GradScaler() if use_amp else None

    # Training Loop
    best_val_metric = 0.0 # Can be accuracy or F1
    patience_counter = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"humor_fusion_{timestamp}"

    print("Starting training loop...")
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp, use_video, use_text) # Pass use_text

        # --- Validation Step (Optional but Recommended) ---
        val_loss, val_acc, val_f1 = -1.0, -1.0, -1.0 # Default values if no validation
        if val_loader:
            val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device, use_video, use_text) # Pass use_text
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Train F1={train_f1:.2f}% | Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, Val F1={val_f1:.2f}%")
            current_metric = val_f1 # Use F1 score for saving best model
        else:
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Train F1={train_f1:.2f}% (No validation)")
            current_metric = train_acc # Use train accuracy if no validation

        # --- Checkpointing and Early Stopping ---
        if current_metric > best_val_metric:
            best_val_metric = current_metric
            patience_counter = 0
            print(f"Saving best model with Metric: {best_val_metric:.2f}%")
            save_path = os.path.join(output_dir, f"{model_name}_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_metric': best_val_metric,
                'config': config # Save config used for this model
            }, save_path)
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop:
                print(f"Early stopping triggered after {args.early_stop} epochs without improvement.")
                break

        if scheduler:
            scheduler.step()

    print(f"\nTraining finished. Best Validation Metric: {best_val_metric:.2f}%")
    print(f"Best model saved to: {os.path.join(output_dir, f'{model_name}_best.pt')}")

if __name__ == "__main__":
    main()
