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

# Import necessary components for the new branches
# Smile Classifier (ResNet based)
try:
    from train_smile import SmileClassifier
    print("Successfully imported SmileClassifier from train_smile")
except ImportError as e:
    print(f"Warning: Could not import SmileClassifier from train_smile: {e}. Smile branch might fail if used.")
    SmileClassifier = None

# Text Humor Classifier (DistilBERT based)
try:
    from train_distil_humor import DistilHumorClassifier
    print("Successfully imported DistilHumorClassifier from train_distil_humor")
except ImportError as e:
    print(f"Warning: Could not import DistilHumorClassifier from train_distil_humor: {e}. Text branch might fail if used.")
    DistilHumorClassifier = None

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
HUMOR_LABELS = ['Non-Humor', 'Humor'] # Updated labels
EMOTION_LABELS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise'] # Example emotion labels
BINARY_LABELS = ['False', 'True'] # For smile, laugh, joke

# --- Custom Collate Function ---
def safe_collate_fn(batch):
    """
    Collate function that filters out None samples returned by the dataset.
    """
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        logging.warning("Batch Collapsed: All samples in this batch failed to load.")
        return None # Signal to skip this batch
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

# --- Model Definition ---

class FusionModel(nn.Module):
    # Updated to accept checkpoints for smile and text models
    def __init__(self, num_classes_humor,
                 hubert_checkpoint_path,
                 smile_checkpoint_path=None, # Optional smile checkpoint
                 text_checkpoint_path=None,  # Optional text checkpoint
                 hubert_model_name="facebook/hubert-base-ls960",
                 smile_model_name="resnet18", # Placeholder, actual model loaded from ckpt
                 text_model_name="distilbert-base-uncased",
                 fusion_dim=512, dropout=0.5,
                 freeze_pretrained=True): # Flag to freeze loaded backbones
        super().__init__()
        self.num_classes_humor = num_classes_humor
        self.freeze_pretrained = freeze_pretrained

        # --- Smile Branch (Visual - ResNet based) ---
        self.smile_model = None
        self.smile_feature_dim = 0
        if smile_checkpoint_path and os.path.exists(smile_checkpoint_path):
            print(f"Loading Smile model weights from checkpoint: {smile_checkpoint_path}")
            if SmileClassifier is None:
                 raise RuntimeError("Smile branch requested but SmileClassifier failed to import.")
            try:
                # Load the Lightning module
                self.smile_model = SmileClassifier.load_from_checkpoint(smile_checkpoint_path)
                # Extract the backbone (remove final classifier)
                if hasattr(self.smile_model, 'model'): # Access the underlying torchvision model
                    self.smile_feature_dim = self.smile_model.model.fc.in_features
                    self.smile_model.model.fc = nn.Identity() # Remove final layer
                    print(f"Smile model loaded. Feature dim: {self.smile_feature_dim}")
                    if self.freeze_pretrained:
                        print("Freezing Smile model parameters.")
                        for param in self.smile_model.parameters():
                            param.requires_grad = False
                    self.smile_model.eval() # Set to eval mode
                else:
                     print("Warning: Could not find 'model' attribute in loaded SmileClassifier. Smile branch disabled.")
                     self.smile_model = None
                     self.smile_feature_dim = 0

            except Exception as e:
                print(f"Error loading Smile checkpoint: {e}. Smile branch disabled.")
                self.smile_model = None
                self.smile_feature_dim = 0
        else:
            print("Smile checkpoint not provided or not found. Smile branch disabled.")

        # --- Laughter Branch (Audio - Hubert SER) ---
        self.hubert_ser = None
        self.hubert_feature_dim = 0
        if hubert_checkpoint_path and os.path.exists(hubert_checkpoint_path):
             print(f"Instantiating HubertSER model ({hubert_model_name})...")
             try:
                 # Instantiate HubertSER - num_classes might differ, handle during loading
                 # Use a dummy num_classes, as the final layer will be replaced anyway
                 self.hubert_ser = HubertSER(hubert_name=hubert_model_name, num_classes=2) # Dummy num_classes=2
                 print(f"Loading Hubert SER weights from checkpoint: {hubert_checkpoint_path}")
                 hub_checkpoint = torch.load(hubert_checkpoint_path, map_location='cpu')
                 # Handle different checkpoint formats
                 if 'model_state_dict' in hub_checkpoint:
                     hub_state_dict = hub_checkpoint['model_state_dict']
                 elif 'state_dict' in hub_checkpoint:
                     hub_state_dict = hub_checkpoint['state_dict']
                     hub_state_dict = {k.replace("model.", ""): v for k, v in hub_state_dict.items()} # Adjust PL keys
                 else:
                     hub_state_dict = hub_checkpoint # Assume raw state dict

                 # Load weights, ignoring final classifier layer
                 load_result_hub = self.hubert_ser.load_state_dict(hub_state_dict, strict=False)
                 print(f"Loaded Hubert SER state_dict. Load result (strict=False): {load_result_hub}")
                 if load_result_hub.missing_keys or load_result_hub.unexpected_keys:
                     print("  Note: Mismatched keys likely due to loading a pre-trained model or different final layer.")
                     # print(f"  Missing keys: {load_result_hub.missing_keys}")
                     # print(f"  Unexpected keys: {load_result_hub.unexpected_keys}")

                 # Get feature dimension *before* the final FC layer of HubertSER
                 self.hubert_feature_dim = self.hubert_ser.fc.in_features # Get dim before replacing
                 self.hubert_ser.fc = nn.Identity() # Remove final layer
                 print(f"Hubert SER model loaded. Feature dim: {self.hubert_feature_dim}")

                 if self.freeze_pretrained:
                     print("Freezing Hubert backbone parameters.")
                     for param in self.hubert_ser.hubert.parameters():
                         param.requires_grad = False
                 self.hubert_ser.eval() # Set to eval mode

             except Exception as e:
                 print(f"Error loading Hubert checkpoint: {e}. Laughter branch disabled.")
                 self.hubert_ser = None
                 self.hubert_feature_dim = 0
        else:
            print("Hubert checkpoint not provided or not found. Laughter branch disabled.")

        # --- Text Humor Branch (DistilBERT based) ---
        self.text_model = None
        self.text_feature_dim = 0
        if text_checkpoint_path and os.path.exists(text_checkpoint_path):
            print(f"Loading Text Humor model weights from checkpoint: {text_checkpoint_path}")
            if DistilHumorClassifier is None:
                 raise RuntimeError("Text branch requested but DistilHumorClassifier failed to import.")
            try:
                # Load the Lightning module
                self.text_model = DistilHumorClassifier.load_from_checkpoint(text_checkpoint_path)
                # Extract the backbone (remove final classifier)
                if hasattr(self.text_model, 'bert'): # Access the underlying transformer model
                    self.text_feature_dim = self.text_model.bert.config.dim
                    self.text_model.classifier = nn.Identity() # Remove final layer
                    self.text_model.dropout = nn.Identity() # Also remove dropout before classifier
                    print(f"Text model loaded. Feature dim: {self.text_feature_dim}")
                    if self.freeze_pretrained:
                        print("Freezing Text model parameters.")
                        for param in self.text_model.parameters():
                            param.requires_grad = False
                    self.text_model.eval() # Set to eval mode
                else:
                     print("Warning: Could not find 'bert' attribute in loaded DistilHumorClassifier. Text branch disabled.")
                     self.text_model = None
                     self.text_feature_dim = 0
            except Exception as e:
                print(f"Error loading Text Humor checkpoint: {e}. Text branch disabled.")
                self.text_model = None
                self.text_feature_dim = 0
        else:
            print("Text Humor checkpoint not provided or not found. Text branch disabled.")


        # --- Cue-Specific Heads (Optional - can predict humor directly from each branch) ---
        # These heads are *separate* from the final fusion MLP
        self.smile_head = nn.Linear(self.smile_feature_dim, 1) if self.smile_model else None
        self.laugh_head = nn.Linear(self.hubert_feature_dim, 1) if self.hubert_ser else None
        self.joke_head = nn.Linear(self.text_feature_dim, 1) if self.text_model else None


        # --- Fusion MLP (Takes concatenated features from active branches) ---
        # Calculate fusion input dimension based on active & loaded modalities
        fusion_input_dim = (self.smile_feature_dim if self.smile_model else 0) + \
                           (self.hubert_feature_dim if self.hubert_ser else 0) + \
                           (self.text_feature_dim if self.text_model else 0)

        print(f"Fusion input dimension: {fusion_input_dim}")
        if fusion_input_dim == 0:
            raise ValueError("No valid pretrained models loaded. Cannot create fusion layer.")

        # Define the MLP layers dynamically based on fusion_dim (example)
        # You might want to make hidden layer sizes configurable via YAML
        hidden_layers = [fusion_dim // 2, fusion_dim // 4] # Example hidden sizes
        layers = []
        current_dim = fusion_input_dim
        for h_dim in hidden_layers:
             layers.append(nn.Linear(current_dim, h_dim))
             layers.append(nn.ReLU())
             layers.append(nn.Dropout(dropout))
             current_dim = h_dim
        layers.append(nn.Linear(current_dim, self.num_classes_humor)) # Final output

        self.humor_fusion_mlp = nn.Sequential(*layers)
        print(f"Humor Fusion MLP created: {self.humor_fusion_mlp}")

    def _pool_text_cls(self, last_hidden_state):
        """Pool text features using the [CLS] token embedding."""
        # Assuming the [CLS] token is always at index 0
        return last_hidden_state[:, 0]

    def forward(self, audio_input_values=None, audio_attention_mask=None,
                image_input=None, # Changed from video_input
                text_input_ids=None, text_attention_mask=None):
        # image_input shape: [B, C, H, W] (for smile model)

        # --- Extract Features from each branch ---
        smile_features = None
        if self.smile_model and image_input is not None:
            # Ensure image_input is on the correct device
            image_input = image_input.to(next(self.smile_model.parameters()).device)
            smile_features = self.smile_model(image_input) # Shape: [B, smile_feature_dim]

        laugh_features = None
        if self.hubert_ser and audio_input_values is not None:
             # Ensure audio inputs are on the correct device
             audio_input_values = audio_input_values.to(next(self.hubert_ser.parameters()).device)
             if audio_attention_mask is not None:
                 audio_attention_mask = audio_attention_mask.to(audio_input_values.device)
             # Pass through Hubert backbone (fc layer already removed)
             laugh_features = self.hubert_ser(input_values=audio_input_values, attention_mask=audio_attention_mask) # Shape: [B, hubert_feature_dim]

        joke_features = None
        if self.text_model and text_input_ids is not None:
             # Ensure text inputs are on the correct device
             text_input_ids = text_input_ids.to(next(self.text_model.parameters()).device)
             if text_attention_mask is not None:
                 text_attention_mask = text_attention_mask.to(text_input_ids.device)
             # Pass through DistilBERT backbone (classifier already removed)
             joke_features = self.text_model(input_ids=text_input_ids, attention_mask=text_attention_mask) # Shape: [B, text_feature_dim]


        # --- Collect active features for fusion ---
        active_features = []
        if smile_features is not None: active_features.append(smile_features)
        if laugh_features is not None: active_features.append(laugh_features)
        if joke_features is not None: active_features.append(joke_features)

        if not active_features:
            # Handle case where no modalities produced features (e.g., all checkpoints failed to load)
            # Return zero logits or raise an error
            print("Warning: No features extracted from any modality.")
            # Create dummy zero logits based on expected batch size (infer from one of the inputs if possible)
            batch_size = audio_input_values.shape[0] if audio_input_values is not None else (image_input.shape[0] if image_input is not None else (text_input_ids.shape[0] if text_input_ids is not None else 1))
            dummy_logits = torch.zeros(batch_size, self.num_classes_humor, device=self.humor_fusion_mlp[-1].weight.device) # Use device of last layer
            # Return dummy outputs for all heads
            return {
                "humor_logits": dummy_logits,
                "smile_logits": None,
                "laugh_logits": None,
                "joke_logits": None
            }


        # Concatenate features
        fused_features = torch.cat(active_features, dim=1)

        # --- Final Humor Prediction ---
        final_humor_logits = self.humor_fusion_mlp(fused_features) # Binary classification

        # --- Optional: Cue-Specific Predictions (if heads exist) ---
        smile_logits = self.smile_head(smile_features) if self.smile_head and smile_features is not None else None
        laugh_logits = self.laugh_head(laugh_features) if self.laugh_head and laugh_features is not None else None
        joke_logits = self.joke_head(joke_features) if self.joke_head and joke_features is not None else None

        # Return all relevant outputs
        return {
            "humor_logits": final_humor_logits, # Main output
            "smile_logits": smile_logits,
            "laugh_logits": laugh_logits,
            "joke_logits": joke_logits
        }


# --- Training and Validation Functions ---
# Updated to handle HumorDataset output format including text
# Pass criteria and loss_weights as arguments
def train_epoch(model, dataloader, criterion_emotion, criterion_binary, loss_weights, optimizer, scaler, device, use_amp, use_video, use_text):
    model.train()
    # Ensure frozen backbones remain in eval mode
    if hasattr(model, 'hubert_ser') and not any(p.requires_grad for p in model.hubert_ser.hubert.parameters()):
        model.hubert_ser.hubert.eval()
    if use_video and hasattr(model, 'video_embedder') and model.video_embedder and hasattr(model.video_embedder, 'video_embedder') and not any(p.requires_grad for p in model.video_embedder.video_embedder.parameters()):
        model.video_embedder.video_embedder.eval()
    if use_text and hasattr(model, 'text_embedder') and model.text_embedder and not any(p.requires_grad for p in model.text_embedder.parameters()):
        model.text_embedder.eval()

    running_loss = 0.0
    # Initialize accumulators for metrics
    all_labels = {'emotion': [], 'humor': [], 'smile': [], 'laugh': [], 'joke': []}
    all_preds = {'emotion': [], 'humor': [], 'smile': [], 'laugh': [], 'joke': []}

    progress_bar = tqdm(dataloader, desc="Training Humor")

    for batch in progress_bar:
        if batch is None:
            continue

        audio_input_values = batch['audio_input_values'].to(device)
        audio_attention_mask = batch['audio_attention_mask'].to(device)

        video_data = None
        if use_video and 'video' in batch:
            video_data = batch['video'].to(device)
            if video_data.dim() == 5 and video_data.shape[2] == 3:
                 video_data = video_data.permute(0, 2, 1, 3, 4)

        text_input_ids = None
        text_attention_mask = None
        if use_text and 'text_input_ids' in batch and 'text_attention_mask' in batch:
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            model_outputs = model(audio_input_values=audio_input_values,
                                  audio_attention_mask=audio_attention_mask,
                                  video_input=video_data,
                                  text_input_ids=text_input_ids,
                                  text_attention_mask=text_attention_mask)

            # --- Calculate Loss (Multi-Task) ---
            total_loss = 0.0
            batch_labels = {}
            batch_logits = {}

            # Fetch labels and logits
            for task in ['emotion', 'humor', 'smile', 'laugh', 'joke']:
                batch_labels[task] = batch.get(f'{task}_label')
                batch_logits[task] = model_outputs.get(f'{task}_logits')

            # Calculate weighted loss for each task where labels and logits are available
            if batch_labels['emotion'] is not None and batch_logits['emotion'] is not None:
                loss_emotion = criterion_emotion(batch_logits['emotion'], batch_labels['emotion'].long().to(device))
                total_loss += loss_weights["emotion"] * loss_emotion
            if batch_labels['humor'] is not None and batch_logits['humor'] is not None:
                loss_humor = criterion_binary(batch_logits['humor'].squeeze(-1), batch_labels['humor'].float().to(device))
                total_loss += loss_weights["humor"] * loss_humor
            if batch_labels['smile'] is not None and batch_logits['smile'] is not None:
                loss_smile = criterion_binary(batch_logits['smile'].squeeze(-1), batch_labels['smile'].float().to(device))
                total_loss += loss_weights["smile"] * loss_smile
            if batch_labels['laugh'] is not None and batch_logits['laugh'] is not None:
                loss_laugh = criterion_binary(batch_logits['laugh'].squeeze(-1), batch_labels['laugh'].float().to(device))
                total_loss += loss_weights["laugh"] * loss_laugh
            if batch_labels['joke'] is not None and batch_logits['joke'] is not None:
                loss_joke = criterion_binary(batch_logits['joke'].squeeze(-1), batch_labels['joke'].float().to(device))
                total_loss += loss_weights["joke"] * loss_joke

            loss = total_loss

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

        # Accumulate predictions and labels for metrics
        with torch.no_grad():
            if batch_labels['emotion'] is not None and batch_logits['emotion'] is not None:
                preds_emotion = torch.argmax(batch_logits['emotion'], dim=1)
                all_preds['emotion'].extend(preds_emotion.cpu().numpy())
                all_labels['emotion'].extend(batch_labels['emotion'].cpu().numpy())
            for task in ['humor', 'smile', 'laugh', 'joke']:
                if batch_labels[task] is not None and batch_logits[task] is not None:
                    preds_binary = (torch.sigmoid(batch_logits[task].squeeze(-1)) > 0.5).long()
                    all_preds[task].extend(preds_binary.cpu().numpy())
                    all_labels[task].extend(batch_labels[task].cpu().numpy())

        progress_bar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(dataloader)
    # Calculate metrics for each task
    metrics = {}
    if all_labels['emotion']:
        metrics['emotion_acc'] = accuracy_score(all_labels['emotion'], all_preds['emotion']) * 100
        metrics['emotion_f1'] = f1_score(all_labels['emotion'], all_preds['emotion'], average='macro', zero_division=0) * 100
    for task in ['humor', 'smile', 'laugh', 'joke']:
        if all_labels[task]:
            metrics[f'{task}_acc'] = accuracy_score(all_labels[task], all_preds[task]) * 100
            metrics[f'{task}_f1'] = f1_score(all_labels[task], all_preds[task], average='binary', zero_division=0) * 100 # Use binary average for these

    # Return primary metrics (e.g., humor F1) along with the full dict
    primary_acc = metrics.get('humor_acc', 0.0)
    primary_f1 = metrics.get('humor_f1', 0.0)

    return epoch_loss, primary_acc, primary_f1, metrics


# Pass criteria and loss_weights as arguments
def validate(model, dataloader, criterion_emotion, criterion_binary, loss_weights, device, use_video, use_text):
    model.eval()
    running_loss = 0.0
    # Initialize accumulators for metrics
    all_labels = {'emotion': [], 'humor': [], 'smile': [], 'laugh': [], 'joke': []}
    all_preds = {'emotion': [], 'humor': [], 'smile': [], 'laugh': [], 'joke': []}
    progress_bar = tqdm(dataloader, desc="Validating Humor")

    with torch.no_grad():
        for batch in progress_bar:
            if batch is None:
                continue
            audio_input_values = batch['audio_input_values'].to(device)
            audio_attention_mask = batch['audio_attention_mask'].to(device)

            video_data = None
            if use_video and 'video' in batch:
                video_data = batch['video'].to(device)
                if video_data.dim() == 5 and video_data.shape[2] == 3:
                    video_data = video_data.permute(0, 2, 1, 3, 4)

            text_input_ids = None
            text_attention_mask = None
            if use_text and 'text_input_ids' in batch and 'text_attention_mask' in batch:
                text_input_ids = batch['text_input_ids'].to(device)
                text_attention_mask = batch['text_attention_mask'].to(device)

            model_outputs = model(audio_input_values=audio_input_values,
                                  audio_attention_mask=audio_attention_mask,
                                  video_input=video_data,
                                  text_input_ids=text_input_ids,
                                  text_attention_mask=text_attention_mask)

            # --- Calculate Loss (Multi-Task) ---
            total_loss = 0.0
            batch_labels = {}
            batch_logits = {}

            # Fetch labels and logits
            for task in ['emotion', 'humor', 'smile', 'laugh', 'joke']:
                batch_labels[task] = batch.get(f'{task}_label')
                batch_logits[task] = model_outputs.get(f'{task}_logits')

            # Calculate weighted loss
            if batch_labels['emotion'] is not None and batch_logits['emotion'] is not None:
                loss_emotion = criterion_emotion(batch_logits['emotion'], batch_labels['emotion'].long().to(device))
                total_loss += loss_weights["emotion"] * loss_emotion
            if batch_labels['humor'] is not None and batch_logits['humor'] is not None:
                loss_humor = criterion_binary(batch_logits['humor'].squeeze(-1), batch_labels['humor'].float().to(device))
                total_loss += loss_weights["humor"] * loss_humor
            if batch_labels['smile'] is not None and batch_logits['smile'] is not None:
                loss_smile = criterion_binary(batch_logits['smile'].squeeze(-1), batch_labels['smile'].float().to(device))
                total_loss += loss_weights["smile"] * loss_smile
            if batch_labels['laugh'] is not None and batch_logits['laugh'] is not None:
                loss_laugh = criterion_binary(batch_logits['laugh'].squeeze(-1), batch_labels['laugh'].float().to(device))
                total_loss += loss_weights["laugh"] * loss_laugh
            if batch_labels['joke'] is not None and batch_logits['joke'] is not None:
                loss_joke = criterion_binary(batch_logits['joke'].squeeze(-1), batch_labels['joke'].float().to(device))
                total_loss += loss_weights["joke"] * loss_joke

            loss = total_loss
            running_loss += loss.item()

            # Accumulate predictions and labels for metrics
            if batch_labels['emotion'] is not None and batch_logits['emotion'] is not None:
                preds_emotion = torch.argmax(batch_logits['emotion'], dim=1)
                all_preds['emotion'].extend(preds_emotion.cpu().numpy())
                all_labels['emotion'].extend(batch_labels['emotion'].cpu().numpy())
            for task in ['humor', 'smile', 'laugh', 'joke']:
                if batch_labels[task] is not None and batch_logits[task] is not None:
                    preds_binary = (torch.sigmoid(batch_logits[task].squeeze(-1)) > 0.5).long()
                    all_preds[task].extend(preds_binary.cpu().numpy())
                    all_labels[task].extend(batch_labels[task].cpu().numpy())

    val_loss = running_loss / len(dataloader)
    # Calculate metrics for each task
    metrics = {}
    print("\n--- Validation Results ---")
    if all_labels['emotion']:
        metrics['emotion_acc'] = accuracy_score(all_labels['emotion'], all_preds['emotion']) * 100
        metrics['emotion_f1'] = f1_score(all_labels['emotion'], all_preds['emotion'], average='macro', zero_division=0) * 100
        print(f"Emotion Acc: {metrics['emotion_acc']:.2f}%, Emotion F1: {metrics['emotion_f1']:.2f}%")
        # print("Emotion Classification Report:")
        # print(classification_report(all_labels['emotion'], all_preds['emotion'], target_names=EMOTION_LABELS, zero_division=0))

    for task, task_labels in [('humor', HUMOR_LABELS), ('smile', BINARY_LABELS), ('laugh', BINARY_LABELS), ('joke', BINARY_LABELS)]:
        if all_labels[task]:
            metrics[f'{task}_acc'] = accuracy_score(all_labels[task], all_preds[task]) * 100
            metrics[f'{task}_f1'] = f1_score(all_labels[task], all_preds[task], average='binary', zero_division=0) * 100
            print(f"{task.capitalize()} Acc: {metrics[f'{task}_acc']:.2f}%, {task.capitalize()} F1: {metrics[f'{task}_f1']:.2f}%")
            print(f"{task.capitalize()} Classification Report:")
            print(classification_report(all_labels[task], all_preds[task], target_names=task_labels, zero_division=0))

    # Return primary metrics (e.g., humor F1) along with the full dict
    primary_acc = metrics.get('humor_acc', 0.0)
    primary_f1 = metrics.get('humor_f1', 0.0)

    return val_loss, primary_acc, primary_f1, metrics

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
    hubert_checkpoint = config['model'].get('hubert_checkpoint', None)
    smile_checkpoint = config['model'].get('smile_checkpoint', None) # Get smile checkpoint path
    text_checkpoint = config['model'].get('text_checkpoint', None)   # Get text checkpoint path
    hubert_model_name = config['model'].get('hubert_model_name', "facebook/hubert-base-ls960")
    text_model_name = config['model'].get('text_model_name', "distilbert-base-uncased")
    freeze_pretrained = config['model'].get('freeze_pretrained', True) # Default to freezing

    # Check if at least one modality is active and has a checkpoint
    if not hubert_checkpoint and not smile_checkpoint and not text_checkpoint:
         print("Error: At least one checkpoint (hubert_checkpoint, smile_checkpoint, or text_checkpoint) must be provided in the config.")
         sys.exit(1)
    if not hubert_checkpoint: print("Warning: Hubert (Laughter) checkpoint not provided. Laughter branch disabled.")
    if not smile_checkpoint: print("Warning: Smile checkpoint not provided. Smile branch disabled.")
    if not text_checkpoint: print("Warning: Text checkpoint not provided. Text branch disabled.")
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
    num_classes_emotion = model_params.get('num_classes_emotion', 6) # Get emotion class count from config, default 6
    model = FusionModel(
        num_classes_humor=model_params.get('num_classes_humor', 2), # Default to 2 (Humor/Non-Humor)
        hubert_checkpoint_path=hubert_checkpoint,
        smile_checkpoint_path=smile_checkpoint,
        text_checkpoint_path=text_checkpoint,
        hubert_model_name=hubert_model_name,
        # smile_model_name is not needed as model is loaded from checkpoint
        text_model_name=text_model_name,
        fusion_dim=model_params.get('fusion_dim', 512),
        dropout=model_params.get('dropout', 0.5),
        freeze_pretrained=freeze_pretrained
    ).to(device)
    print("Fusion Model created.")

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
    # Criterion for binary humor classification (expects logits)
    # Use BCEWithLogitsLoss for numerical stability
    criterion_humor = nn.BCEWithLogitsLoss()
    # Optional: Define criteria for auxiliary heads if you train them too
    criterion_aux = nn.BCEWithLogitsLoss() if model.smile_head or model.laugh_head or model.joke_head else None

    scaler = GradScaler() if use_amp else None

    # Loss weights (fetch from config or use defaults)
    # Focus primarily on the main humor prediction head
    loss_weights = {
        "humor": config['training'].get('loss_weight_humor', 1.0),     # Main head
        "laugh": config['training'].get('loss_weight_laugh', 0.1),     # Aux laugh head
        "smile": config['training'].get('loss_weight_smile', 0.1),     # Aux smile head
        "joke": config['training'].get('loss_weight_joke', 0.1),       # Aux joke head
    }
    # Disable weights for heads that don't exist
    if not model.smile_head: loss_weights['smile'] = 0.0
    if not model.laugh_head: loss_weights['laugh'] = 0.0
    if not model.joke_head: loss_weights['joke'] = 0.0
    print(f"Using loss weights: {loss_weights}")

    # Training Loop
    best_val_metric = 0.0 # Can be accuracy or F1
    patience_counter = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"humor_fusion_{timestamp}"

    print("Starting training loop...")
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        # Pass humor criterion and aux criterion (if used)
        train_loss, train_acc, train_f1, train_metrics = train_epoch(
            model, train_loader, criterion_humor, criterion_aux, loss_weights,
            optimizer, scaler, device, use_amp
        )

        # --- Validation Step (Optional but Recommended) ---
        val_loss, val_acc, val_f1 = -1.0, -1.0, -1.0 # Default values if no validation
        val_metrics = {}
        if val_loader:
            val_loss, val_acc, val_f1, val_metrics = validate(
                model, val_loader, criterion_humor, criterion_aux, loss_weights, device
            )
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Humor Acc={train_acc:.2f}%, Train Humor F1={train_f1:.2f}% | Val Loss={val_loss:.4f}, Val Humor Acc={val_acc:.2f}%, Val Humor F1={val_f1:.2f}%")
            # Print aux metrics if available
            for task in ['smile', 'laugh', 'joke']:
                 if f'{task}_acc' in train_metrics: print(f"  Train {task.capitalize()} Acc: {train_metrics[f'{task}_acc']:.2f}%", end='')
                 if f'{task}_f1' in train_metrics: print(f" F1: {train_metrics[f'{task}_f1']:.2f}%", end='; ')
            print()
            for task in ['smile', 'laugh', 'joke']:
                 if f'{task}_acc' in val_metrics: print(f"  Val {task.capitalize()} Acc: {val_metrics[f'{task}_acc']:.2f}%", end='')
                 if f'{task}_f1' in val_metrics: print(f" F1: {val_metrics[f'{task}_f1']:.2f}%", end='; ')
            print()

            current_metric = val_f1 # Use validation F1 score for saving best model
        else:
            # Only training metrics available
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Humor Acc={train_acc:.2f}%, Train Humor F1={train_f1:.2f}% (No validation)")
            for task in ['smile', 'laugh', 'joke']:
                 if f'{task}_acc' in train_metrics: print(f"  Train {task.capitalize()} Acc: {train_metrics[f'{task}_acc']:.2f}%", end='')
                 if f'{task}_f1' in train_metrics: print(f" F1: {train_metrics[f'{task}_f1']:.2f}%", end='; ')
            print()
            current_metric = train_f1 # Use train F1 if no validation

        # --- Checkpointing and Early Stopping ---
        if current_metric > best_val_metric:
            best_val_metric = current_metric
            patience_counter = 0
            print(f"Saving best model with Humor F1: {best_val_metric:.2f}%")
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

    print(f"\nTraining finished. Best Validation Humor F1: {best_val_metric:.2f}%")
    print(f"Best model saved to: {os.path.join(output_dir, f'{model_name}_best.pt')}")

if __name__ == "__main__":
    main()
