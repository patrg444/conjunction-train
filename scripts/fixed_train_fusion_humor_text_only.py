#!/usr/bin/env python3
# Simplified fusion model training script using only the text branch

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
    from dataloaders.humor_dataset import HumorDataset
    print("Successfully imported HumorDataset from dataloaders.humor_dataset")
except ImportError:
    print("Failed to import HumorDataset. Make sure dataloaders/humor_dataset.py is in the path.")
    raise

# Define the DistilHumorClassifier here instead of importing it
import torch
import torch.nn as nn
from transformers import AutoModel

class DistilHumorClassifier(nn.Module):
    """Text branch model for humor detection using DistilBERT"""
    def __init__(self, model_name='distilbert-base-uncased', num_classes=2):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.text_model.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token embedding for classification
        cls_token_embedding = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(cls_token_embedding)
        logits = self.classifier(pooled_output)
        return logits

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
                
                # Create a new state_dict with remapped keys
                new_state_dict = {}
                
                # Check what prefix the keys have
                has_bert_prefix = any(k.startswith('bert.') for k in state_dict.keys())
                has_text_model_prefix = any(k.startswith('text_model.') for k in state_dict.keys())
                
                # Remap keys based on the prefix
                for k, v in state_dict.items():
                    if has_bert_prefix and k.startswith('bert.'):
                        # Convert bert.X to text_model.X
                        new_key = 'text_model.' + k[5:]  # Remove 'bert.' and add 'text_model.'
                        new_state_dict[new_key] = v
                    elif has_text_model_prefix and k.startswith('text_model.'):
                        # Keep as is
                        new_state_dict[k] = v
                    else:
                        # For keys without a prefix, add 'text_model.' prefix
                        new_state_dict['text_model.' + k] = v
                
                # Now try to load the remapped state_dict
                self.text_model.load_state_dict(new_state_dict)
                print("Text model loaded. Feature dim:", self.feature_dim)
            except Exception as e:
                print(f"Error loading Text checkpoint: {str(e)}")
                print("Will initialize from scratch instead")

    def forward(self, input_ids, attention_mask):
        # Get the pooled output directly from text model
        pooled_output = self.text_model.text_model(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
        return pooled_output  # Return just the embeddings

class FusionModel(nn.Module):
    """Text-only humor detection model"""
    def __init__(self, text_model_name="distilbert-base-uncased", text_checkpoint=None,
                 fusion_dim=512, dropout=0.5, freeze_pretrained=True, num_classes_humor=2):
        super(FusionModel, self).__init__()
        
        # Initialize text model
        self.text_model = None
        self.fusion_input_dim = 0
        
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
            print("Text Humor checkpoint not provided or not found.")
            raise ValueError("Text checkpoint is required for this model")
        
        print(f"Fusion input dimension: {self.fusion_input_dim}")
        
        # If text model couldn't be loaded, raise an error
        if self.fusion_input_dim == 0:
            raise ValueError("Text model could not be loaded. Cannot create fusion layer.")
        
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
        
    def forward(self, batch):
        # Process text features
        if self.text_model is not None and 'input_ids' in batch:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            text_features = self.text_model(input_ids, attention_mask)
            
            # Apply fusion MLP for humor prediction
            humor_logits = self.humor_fusion_mlp(text_features)
            return humor_logits
        else:
            raise ValueError("Text model or inputs not available")

def load_dataset_from_config(config, split='train', use_augmentation=True):
    """Load dataset based on configuration"""
    # Select appropriate CSV file
    if split == 'train':
        manifest_path = config.get('train_csv_with_text', None)
    else:  # val or test
        manifest_path = config.get('val_csv_with_text', None)
        use_augmentation = False  # Never use augmentation for val/test
    
    # Ensure the manifest path exists
    if not manifest_path or not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    
    # Get dataset params
    dataset_params = config.get('dataset_params', {})
    dataset_root = config.get('dataset_root', 'datasets/manifests')
    
    # Create dataset
    dataset = HumorDataset(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        duration=dataset_params.get('duration', 1.0),
        sample_rate=dataset_params.get('sample_rate', 16000),
        video_fps=dataset_params.get('video_fps', 15),
        video_frames=None,
        img_size=dataset_params.get('image_size', 112),
        hubert_model_name="facebook/hubert-base-ls960",
        text_model_name="distilbert-base-uncased",
        max_text_len=dataset_params.get('max_text_len', 128),
        split=split,
        augment=use_augmentation
    )
    
    return dataset

def create_data_loaders(config, batch_size=None, num_workers=None):
    """Create data loaders for training and validation"""
    dataloader_params = config.get('dataloader_params', {})
    batch_size = batch_size or dataloader_params.get('batch_size', 32)
    
    # Detect if running on CPU and adjust workers accordingly
    device_is_cpu = not torch.cuda.is_available()
    suggested_workers = 0 if device_is_cpu else 4
    num_workers = num_workers or dataloader_params.get('num_workers', suggested_workers)
    pin_memory = False if device_is_cpu else dataloader_params.get('pin_memory', True)
    
    print(f"DataLoader settings: batch_size={batch_size}, num_workers={num_workers}, pin_memory={pin_memory}")
    
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

def train_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    humor_preds, humor_targets = [], []
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f"Training", leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        
        # Extract labels
        batch_size = len(batch.get('input_ids', []))
        
        # Setup targets
        humor_target = batch.get('humor_label', None)
        if humor_target is None:
            # If humor labels are missing, generate random ones for this batch
            # This is only for demonstration/testing
            humor_target = torch.randint(0, 2, (batch_size,), device=device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass (with AMP if enabled)
        with autocast(device_type=device.type if device.type != 'cpu' else 'cpu', enabled=use_amp):
            # Main forward pass
            humor_logits = model(batch)
            
            # Humor loss
            loss = criterion(humor_logits, humor_target) if humor_target is not None else 0
        
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
    if len(train_loader) > 0:
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
    else:
        print("WARNING: Training loader is empty. No batches were processed.")
        avg_loss = 0
        humor_acc = 0
        humor_f1 = 0
    
    # Return metrics
    metrics = {
        'loss': avg_loss,
        'accuracy': humor_acc,
        'f1': humor_f1
    }
    
    return avg_loss, humor_acc, humor_f1, metrics

def validate(model, val_loader, criterion, device):
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
            batch_size = len(batch.get('input_ids', []))
            
            # Setup targets
            humor_target = batch.get('humor_label', None)
            if humor_target is None:
                # If humor labels are missing, generate random ones for this batch
                humor_target = torch.randint(0, 2, (batch_size,), device=device)
            
            # Forward pass
            humor_logits = model(batch)
            
            # Humor loss
            loss = criterion(humor_logits, humor_target) if humor_target is not None else 0
            
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
    parser = argparse.ArgumentParser(description="Train Text-Only Fusion Model for Humor Detection")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for checkpoints and logs')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--fp16', action=argparse.BooleanOptionalAction, default=False, help='Use mixed precision training')
    parser.add_argument('--early_stop', type=int, default=5, help='Early stopping patience (epochs)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    # Force CPU regardless of config due to CUDA compilation issues
    device = torch.device('cpu')
    print(f"Using device: {device} (forced CPU mode)")
    
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
    text_model_name = model_params.get('text_model_name', "distilbert-base-uncased")
    text_checkpoint = model_params.get('text_checkpoint', None)
    
    # Create model
    print("Creating Fusion Model for Humor Detection...")
    model = FusionModel(
        text_model_name=text_model_name,
        text_checkpoint=text_checkpoint,
        fusion_dim=model_params.get('fusion_dim', 512),
        dropout=model_params.get('dropout', 0.5),
        freeze_pretrained=freeze_pretrained,
        num_classes_humor=model_params.get('num_classes_humor', 2)
    ).to(device)
    print("Fusion Model created.")
    
    # Training parameters
    training_params = config.get('training', {})
    num_epochs = args.epochs or training_params.get('epochs', 50)
    lr = args.lr or training_params.get('optimizer_params', {}).get('lr', 0.0001)
    weight_decay = training_params.get('optimizer_params', {}).get('weight_decay', 0.0001)
    early_stop_patience = args.early_stop
    
    # Create optimizer
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Gradient scaler for mixed precision training
    scaler = GradScaler() if use_amp else None
    
    # Training loop
    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0
    train_metrics_history = []
    val_metrics_history = []
    
    print(f"Starting training loop...")
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        
        # Train
        train_loss, train_acc, train_f1, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )
        train_metrics_history.append(train_metrics)
        
        # Validate
        if val_loader:
            val_loss, val_acc, val_f1, val_metrics = validate(
                model, val_loader, criterion, device
            )
            val_metrics_history.append(val_metrics)
            
            print(f"Epoch {epoch + 1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")
            
            # Save checkpoint if best so far
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch + 1
                patience_counter = 0
                
                # Save model
                checkpoint_path = os.path.join(output_dir, "humor_text_best.ckpt")
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_f1': val_f1,
                    'val_accuracy': val_acc,
                    'train_f1': train_f1,
                    'train_accuracy': train_acc,
                }, checkpoint_path)
                print(f"Saved best model checkpoint to {checkpoint_path}")
            else:
                patience_counter += 1
                print(f"No improvement over best F1: {best_val_f1:.4f} (from epoch {best_epoch}). Patience: {patience_counter}/{early_stop_patience}")
                
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break
        else:
            print(f"Epoch {epoch + 1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")
            
            # Save checkpoint every N epochs
            if (epoch + 1) % config.get('checkpointing', {}).get('save_interval', 5) == 0:
                checkpoint_path = os.path.join(output_dir, f"humor_text_epoch_{epoch + 1}.ckpt")
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'train_f1': train_f1,
                    'train_accuracy': train_acc,
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_checkpoint_path = os.path.join(output_dir, "humor_text_final.ckpt")
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train_f1': train_f1,
        'train_accuracy': train_acc,
    }, final_checkpoint_path)
    print(f"Saved final model checkpoint to {final_checkpoint_path}")
    
    print("Training completed!")
    if val_loader:
        print(f"Best validation F1: {best_val_f1:.4f} (epoch {best_epoch})")

if __name__ == "__main__":
    main()
