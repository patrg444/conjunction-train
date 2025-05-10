#!/usr/bin/env python3
# Fixed fusion model training script with safe handling of missing models

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import yaml
import logging
import random
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"Added project root to sys.path: {project_root}")

# Import modules from project
try:
    from train_smile import SmileClassifier
    print("Successfully imported SmileClassifier from train_smile")
except ImportError:
    print("Failed to import SmileClassifier. Make sure train_smile.py is in the path.")
    SmileClassifier = None

try:
    from train_distil_humor import DistilHumorClassifier
    print("Successfully imported DistilHumorClassifier from train_distil_humor")
except ImportError:
    print("Failed to import DistilHumorClassifier. Make sure train_distil_humor.py is in the path.")
    DistilHumorClassifier = None

try:
    from ser_hubert.hubert_ser_module import HubertSER
    print("Successfully imported HubertSER from ser_hubert.hubert_ser_module")
except ImportError:
    print("Failed to import HubertSER. Make sure ser_hubert module is in path.")
    HubertSER = None

try:
    from dataloaders.humor_dataset import HumorDataset
    print("Successfully imported HumorDataset from dataloaders.humor_dataset")
except ImportError:
    print("Failed to import HumorDataset. Make sure dataloaders/humor_dataset.py is in the path.")
    raise

class TextBranchModel(nn.Module):
    """Text branch model for humor detection using DistilBERT"""
    def __init__(self, checkpoint_path=None, model_name="distilbert-base-uncased", num_classes=2):
        super(TextBranchModel, self).__init__()
        self.text_model = DistilHumorClassifier(model_name=model_name, num_classes=num_classes)
        self.feature_dim = self.text_model.text_model.config.hidden_size

        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading Text Humor model weights from checkpoint: {checkpoint_path}")
            try:
                # Try to load with weights_only=False first (legacy support)
                state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                
                # Remove 'text_model.' prefix if present in keys
                if all(k.startswith('text_model.') for k in state_dict.keys()):
                    state_dict = {k.replace('text_model.', ''): v for k, v in state_dict.items()}
                    
                self.text_model.load_state_dict(state_dict)
                print("Text model loaded. Feature dim:", self.feature_dim)
            except Exception as e:
                print(f"Error loading Text checkpoint: {str(e)}")
                raise

    def forward(self, input_ids, attention_mask):
        # Get the pooled output directly from text model
        pooled_output = self.text_model.text_model(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
        return pooled_output  # Return just the embeddings

class SmileBranchModel(nn.Module):
    """Smile branch model for humor detection using a ResNet on face images"""
    def __init__(self, checkpoint_path=None, num_classes=2):
        super(SmileBranchModel, self).__init__()
        self.smile_model = SmileClassifier(num_classes=num_classes)
        self.feature_dim = 512  # ResNet18 final feature dimension

        # Load checkpoint if provided and exists
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading Smile model weights from checkpoint: {checkpoint_path}")
            try:
                # Try different loading approaches
                try:
                    # Try to load with weights_only=False first
                    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                except:
                    # If that fails, try without the weights_only parameter
                    state_dict = torch.load(checkpoint_path, map_location='cpu')
                
                if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                
                # Remove 'model.' prefix if present in keys
                if all(k.startswith('model.') for k in state_dict.keys()):
                    state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
                
                self.smile_model.model.load_state_dict(state_dict)
                print("Smile model loaded. Feature dim:", self.feature_dim)
            except Exception as e:
                print(f"Error loading Smile checkpoint: {str(e)}. Smile branch disabled.")
                raise

    def forward(self, images):
        # Use the backbone to get features, excluding the classification head
        features = self.smile_model.model.backbone(images)
        # Apply global pooling if needed
        if len(features.shape) > 2:  # If features have spatial dimensions
            features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)
        return features

class HubertBranchModel(nn.Module):
    """HuBERT-based model for laughter detection in audio"""
    def __init__(self, checkpoint_path=None, model_name="facebook/hubert-base-ls960", num_classes=2):
        super(HubertBranchModel, self).__init__()
        print(f"Instantiating HubertSER model ({model_name})...")
        self.hubert_ser = HubertSER(model_name=model_name, num_classes=num_classes)
        self.feature_dim = 768  # HuBERT base model feature dimension

        # Load checkpoint if provided and exists
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading Hubert SER weights from checkpoint: {checkpoint_path}")
            try:
                # Try different loading approaches
                try:
                    # Try to load with weights_only=False first (legacy support)
                    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                except:
                    # If that fails, try without the weights_only parameter
                    state_dict = torch.load(checkpoint_path, map_location='cpu')
                
                if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                
                # Try to load the state dict
                self.hubert_ser.load_state_dict(state_dict)
                print("Hubert SER model loaded. Feature dim:", self.feature_dim)
            except Exception as e:
                print(f"Error loading Hubert checkpoint: {str(e)}. Laughter branch disabled.")
                raise
    
    def forward(self, audio_inputs, audio_attention_mask=None):
        # Extract features from the HuBERT model
        outputs = self.hubert_ser.hubert(input_values=audio_inputs, attention_mask=audio_attention_mask)
        # Get the pooled representation (mean of sequence)
        pooled_output = torch.mean(outputs.last_hidden_state, dim=1)
        return pooled_output

class FusionModel(nn.Module):
    """Multimodal fusion model for humor detection"""
    def __init__(self, num_classes_humor=2, 
                 hubert_model_name="facebook/hubert-base-ls960", 
                 hubert_checkpoint=None,
                 smile_checkpoint=None, 
                 text_model_name="distilbert-base-uncased",
                 text_checkpoint=None,
                 fusion_dim=512, dropout=0.5, freeze_pretrained=True):
        super(FusionModel, self).__init__()
        
        # Initialize branch models to None
        self.hubert_ser = None
        self.smile_model = None
        self.text_model = None
        self.fusion_input_dim = 0
        
        # Try to load HuBERT branch if checkpoint provided
        if hubert_checkpoint:
            try:
                self.hubert_ser = HubertBranchModel(
                    checkpoint_path=hubert_checkpoint,
                    model_name=hubert_model_name,
                    num_classes=2  # Laughter/No laughter
                )
                self.fusion_input_dim += self.hubert_ser.feature_dim
                if freeze_pretrained:
                    for param in self.hubert_ser.hubert_ser.hubert.parameters():
                        param.requires_grad = False
                    print("Freezing HuBERT parameters.")
            except Exception as e:
                print(f"Error initializing HuBERT branch: {str(e)}")
                self.hubert_ser = None
        else:
            print("Hubert checkpoint not provided or not found. Laughter branch disabled.")
        
        # Try to load Smile branch if checkpoint provided
        if smile_checkpoint:
            try:
                self.smile_model = SmileBranchModel(
                    checkpoint_path=smile_checkpoint,
                    num_classes=2  # Smile/No smile
                )
                self.fusion_input_dim += self.smile_model.feature_dim
                if freeze_pretrained:
                    for param in self.smile_model.smile_model.model.backbone.parameters():
                        param.requires_grad = False
                    print("Freezing Smile model parameters.")
            except Exception as e:
                print(f"Error initializing Smile branch: {str(e)}")
                self.smile_model = None
        else:
            print("Smile checkpoint not provided or not found. Smile branch disabled.")
            
        # Try to load Text branch if checkpoint provided
        if text_checkpoint:
            try:
                self.text_model = TextBranchModel(
                    checkpoint_path=text_checkpoint,
                    model_name=text_model_name,
                    num_classes=2  # Joke/Non-joke text
                )
                self.fusion_input_dim += self.text_model.feature_dim
                if freeze_pretrained:
                    for param in self.text_model.text_model.text_model.parameters():
                        param.requires_grad = False
                    print("Freezing Text model parameters.")
            except Exception as e:
                print(f"Error initializing Text branch: {str(e)}")
                self.text_model = None
        else:
            print("Text Humor checkpoint not provided or not found. Text branch disabled.")
        
        print(f"Fusion input dimension: {self.fusion_input_dim}")
        
        # If no valid models were loaded, we can't create a fusion layer
        if self.fusion_input_dim == 0:
            raise ValueError("No valid pretrained models loaded. Cannot create fusion layer.")
        
        # Create the fusion MLP
        self.humor_fusion_mlp = nn.Sequential(
            nn.Linear(self.fusion_input_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_classes_humor)
        )
        
        print("Humor Fusion MLP created:", self.humor_fusion_mlp)
        
    def forward(self, batch, use_video=True, use_text=True):
        features = []
        
        # Get features from branch models, if available and enabled
        # Process audio (laughter) features if model is available
        if self.hubert_ser is not None and 'audio_inputs' in batch:
            audio_inputs = batch['audio_inputs']
            audio_attention_mask = batch.get('audio_attention_mask', None)
            audio_features = self.hubert_ser(audio_inputs, audio_attention_mask)
            features.append(audio_features)
        
        # Process video (smile) features if model is available and video use is enabled
        if self.smile_model is not None and use_video and 'images' in batch:
            images = batch['images']
            video_features = self.smile_model(images)
            features.append(video_features)
        
        # Process text features if model is available and text use is enabled
        if self.text_model is not None and use_text and 'input_ids' in batch:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            text_features = self.text_model(input_ids, attention_mask)
            features.append(text_features)
        
        # Concatenate all available features
        if not features:
            raise ValueError("No features available. Check inputs and enabled branches.")
        
        if len(features) == 1:
            # If only one feature type, use it directly
            fused_features = features[0]
        else:
            # Concatenate multiple feature types
            fused_features = torch.cat(features, dim=1)
        
        # Apply fusion MLP for humor prediction
        humor_logits = self.humor_fusion_mlp(fused_features)
        
        # Return the logits (and any auxiliary outputs)
        return humor_logits
        
def load_dataset_from_config(config, split='train', use_augmentation=True):
    """Load dataset based on configuration"""
    # Select appropriate CSV file
    if split == 'train':
        csv_path = config.get('train_csv_with_text', None)
    else:  # val or test
        csv_path = config.get('val_csv_with_text', None)
        use_augmentation = False  # Never use augmentation for val/test
    
    # Ensure the manifest path exists
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"Manifest file not found: {csv_path}")
    
    # Get dataset params
    dataset_params = config.get('dataset_params', {})
    
    # Create dataset
    dataset = HumorDataset(
        csv_path=csv_path,
        use_audio=True,  # Always try to load audio
        use_video=True,  # Always try to load video
        use_text=True,   # Always try to load text
        duration=dataset_params.get('duration', 1.0),
        sample_rate=dataset_params.get('sample_rate', 16000),
        video_fps=dataset_params.get('video_fps', 15),
        max_text_len=dataset_params.get('max_text_len', 128),
        image_size=dataset_params.get('image_size', 112),
        split=split,
        augmentation=use_augmentation
    )
    
    return dataset

def create_data_loaders(config, batch_size=None, num_workers=None):
    """Create data loaders for training and validation"""
    dataloader_params = config.get('dataloader_params', {})
    batch_size = batch_size or dataloader_params.get('batch_size', 32)
    num_workers = num_workers or dataloader_params.get('num_workers', 4)
    pin_memory = dataloader_params.get('pin_memory', True)
    
    # Load training dataset
    train_dataset = load_dataset_from_config(config, split='train', use_augmentation=True)
    print(f"Training dataset: {len(train_dataset)} samples")
    
    # Create training dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    # Check if validation manifest exists
    val_csv = config.get('val_csv_with_text', None)
    if val_csv and os.path.exists(val_csv):
        print(f"Validation manifest found: {val_csv}")
        val_dataset = load_dataset_from_config(config, split='val', use_augmentation=False)
        print(f"Validation dataset: {len(val_dataset)} samples")
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    else:
        print("No validation manifest found. Skipping validation.")
        val_loader = None
    
    return train_loader, val_loader

def train_epoch(model, train_loader, criterion_humor, criterion_aux, loss_weights, 
                optimizer, scaler, device, use_amp, use_video, use_text):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    humor_preds, humor_targets = [], []
    
    # Track metrics for each branch if available
    laugh_preds, laugh_targets = [], []
    smile_preds, smile_targets = [], []
    joke_preds, joke_targets = [], []
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f"Training", leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        
        # Extract labels
        batch_size = len(batch.get('input_ids', batch.get('images', batch.get('audio_inputs', []))))
        
        # Setup targets
        humor_target = batch.get('humor_label', None)
        if humor_target is None:
            # If humor labels are missing, generate random ones for this batch
            # This is only for demonstration/testing
            humor_target = torch.randint(0, 2, (batch_size,), device=device)
        
        # Optional auxiliary targets
        laugh_target = batch.get('laugh_label', None)
        smile_target = batch.get('smile_label', None)
        joke_target = batch.get('joke_label', None)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass (with AMP if enabled)
        with autocast(device_type=device.type if device.type != 'cpu' else 'cpu', enabled=use_amp):
            # Main forward pass
            humor_logits = model(batch, use_video=use_video, use_text=use_text)
            
            # Humor loss (main task)
            humor_loss = criterion_humor(humor_logits, humor_target) if humor_target is not None else 0
            
            # Initialize total loss with weighted humor loss
            loss = loss_weights.get('humor', 1.0) * humor_loss
            
            # Add auxiliary losses if enabled and targets available
            # These depend on model internals and may not be always available
            # We're using getattr to safely check if the model has these branches
            
            # Laughter branch loss
            if (hasattr(model, 'hubert_ser') and model.hubert_ser is not None and 
                laugh_target is not None and loss_weights.get('laugh', 0) > 0):
                
                # Skip if all parameters are frozen
                if any(p.requires_grad for p in model.hubert_ser.hubert_ser.parameters()):
                    # Extract laughter logits if available (depends on model internal structure)
                    if hasattr(model.hubert_ser.hubert_ser, 'get_logits'):
                        laugh_logits = model.hubert_ser.hubert_ser.get_logits(batch['audio_inputs'])
                        laugh_loss = criterion_aux(laugh_logits, laugh_target)
                        loss += loss_weights.get('laugh', 0) * laugh_loss
            
            # Smile branch loss
            if (hasattr(model, 'smile_model') and model.smile_model is not None and 
                smile_target is not None and loss_weights.get('smile', 0) > 0):
                
                # Skip if all parameters are frozen
                if any(p.requires_grad for p in model.smile_model.smile_model.parameters()):
                    # Extract smile logits if available
                    if hasattr(model.smile_model.smile_model, 'get_logits'):
                        smile_logits = model.smile_model.smile_model.get_logits(batch['images'])
                        smile_loss = criterion_aux(smile_logits, smile_target)
                        loss += loss_weights.get('smile', 0) * smile_loss
            
            # Text (joke) branch loss 
            if (hasattr(model, 'text_model') and model.text_model is not None and 
                joke_target is not None and loss_weights.get('joke', 0) > 0):
                
                # Skip if all parameters are frozen
                if any(p.requires_grad for p in model.text_model.text_model.parameters()):
                    # Extract joke logits if available
                    if hasattr(model.text_model.text_model, 'get_logits'):
                        joke_logits = model.text_model.text_model.get_logits(
                            batch['input_ids'], batch['attention_mask'])
                        joke_loss = criterion_aux(joke_logits, joke_target)
                        loss += loss_weights.get('joke', 0) * joke_loss
        
        # Backward pass with AMP if enabled
        if use_amp and device.type != 'cpu':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Track loss and predictions
        total_loss += loss.item()
        
        # Track humor predictions for metrics
        humor_probs = F.softmax(humor_logits, dim=1)
        humor_pred = torch.argmax(humor_probs, dim=1).cpu().numpy()
        humor_preds.extend(humor_pred)
        humor_targets.extend(humor_target.cpu().numpy())
        
        # Update progress bar
        pbar.set_description(f"Train Loss: {loss.item():.4f}")
    
    # Calculate metrics
    avg_loss = total_loss / len(train_loader)
    
    # Calculate accuracy and F1 score for the main humor task
    humor_preds = np.array(humor_preds)
    humor_targets = np.array(humor_targets)
    
    if len(np.unique(humor_targets)) > 1:  # Only if we have both classes
        humor_acc = accuracy_score(humor_targets, humor_preds)
        humor_f1 = f1_score(humor_targets, humor_preds, average='weighted')
    else:
        humor_acc = 0
        humor_f1 = 0
    
    # Return metrics
    metrics = {
        'loss': avg_loss,
        'accuracy': humor_acc,
        'f1': humor_f1
    }
    
    return avg_loss, humor_acc, humor_f1, metrics

def validate(model, val_loader, criterion_humor, device, use_video, use_text):
    """Validate model on validation set"""
    model.eval()
    total_loss = 0
    humor_preds, humor_targets = [], []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validating", leave=False):
            # Move batch to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            # Extract labels
            batch_size = len(batch.get('input_ids', batch.get('images', batch.get('audio_inputs', []))))
            
            # Setup targets
            humor_target = batch.get('humor_label', None)
            if humor_target is None:
                # If humor labels are missing, generate random ones for this batch
                # This is only for demonstration/testing
                humor_target = torch.randint(0, 2, (batch_size,), device=device)
            
            # Forward pass
            humor_logits = model(batch, use_video=use_video, use_text=use_text)
            
            # Humor loss (main task)
            loss = criterion_humor(humor_logits, humor_target) if humor_target is not None else 0
            
            # Track loss and predictions
            total_loss += loss.item()
            
            # Track humor predictions for metrics  
            humor_probs = F.softmax(humor_logits, dim=1)
            humor_pred = torch.argmax(humor_probs, dim=1).cpu().numpy()
            humor_preds.extend(humor_pred)
            humor_targets.extend(humor_target.cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(val_loader)
    
    # Calculate accuracy and F1 score for the main humor task
    humor_preds = np.array(humor_preds)
    humor_targets = np.array(humor_targets)
    
    if len(np.unique(humor_targets)) > 1:  # Only if we have both classes
        humor_acc = accuracy_score(humor_targets, humor_preds)
        humor_f1 = f1_score(humor_targets, humor_preds, average='weighted')
        precision = precision_score(humor_targets, humor_preds, average='weighted')
        recall = recall_score(humor_targets, humor_preds, average='weighted')
    else:
        humor_acc = 0
        humor_f1 = 0
        precision = 0
        recall = 0
    
    # Return metrics
    metrics = {
        'val_loss': avg_loss,
        'val_accuracy': humor_acc,
        'val_f1': humor_f1,
        'val_precision': precision,
        'val_recall': recall
    }
    
    return avg_loss, humor_acc, humor_f1, metrics

def main():
    """Main training function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train Fusion Model for Humor Detection")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for checkpoints and logs')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--use_video', action=argparse.BooleanOptionalAction, default=True, help='Use video branch')
    parser.add_argument('--use_text', action=argparse.BooleanOptionalAction, default=True, help='Use text branch')
    parser.add_argument('--fp16', action=argparse.BooleanOptionalAction, default=False, help='Use mixed precision training')
    parser.add_argument('--early_stop', type=int, default=5, help='Early stopping patience (epochs)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device_str = config.get('training', {}).get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)
    print(f"Using device: {device}")
    
    # Flag for using video or text branches
    use_video = args.use_video
    use_text = args.use_text
    print(f"Using Video: {use_video}")
    print(f"Using Text: {use_text}")
    
    # Mixed precision flag
    use_amp = args.fp16
    print(f"Using Mixed Precision (AMP): {use_amp}")
    
    # Setup output directory
    output_dir = args.output_dir or config.get('checkpointing', {}).get('checkpoint_dir', 'checkpoints/humor_training')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create data loaders
    print("Creating Humor Datasets...")
    data_config = config.get('data', {})
    train_loader, val_loader = create_data_loaders(
        data_config, 
        batch_size=args.batch_size,
        num_workers=data_config.get('dataloader_params', {}).get('num_workers', 4)
    )
    print("Datasets created.")
    
    # Model parameters
    model_params = config.get('model', {})
    freeze_pretrained = model_params.get('freeze_pretrained', True)
    
    # Get model checkpoint paths
    hubert_model_name = model_params.get('hubert_model_name', "facebook/hubert-base-ls960")
    hubert_checkpoint = model_params.get('hubert_checkpoint', None)
    smile_checkpoint = model_params.get('smile_checkpoint', None) 
    text_model_name = model_params.get('text_model_name', "distilbert-
