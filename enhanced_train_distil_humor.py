#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Text-based Humor Classifier using DistilBERT
This script trains a text-based humor detection model using the DistilBERT model
and the transcript text in the humor manifest files.
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('text_humor_training.log')
    ]
)
logger = logging.getLogger(__name__)

class HumorTextDataset(Dataset):
    """Dataset for humor text classification."""
    
    def __init__(self, csv_file, tokenizer, max_length=128):
        """
        Args:
            csv_file (str): Path to the CSV manifest file with text data.
            tokenizer: HuggingFace tokenizer for text processing.
            max_length (int): Maximum sequence length for tokenization.
        """
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Verify required columns exist
        required_cols = ['transcript', 'label']
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"Required columns missing from manifest: {missing}")
            
        # Check for NaN values in transcript column
        if self.df['transcript'].isna().any():
            logger.warning(f"Found {self.df['transcript'].isna().sum()} rows with NaN transcripts. These will be replaced with empty strings.")
            self.df['transcript'].fillna("", inplace=True)
            
        logger.info(f"Loaded manifest from {csv_file} with {len(self.df)} samples")
        logger.info(f"Label distribution: {self.df['label'].value_counts().to_dict()}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text = str(self.df.iloc[idx]['transcript'])
        label = float(self.df.iloc[idx]['label'])  # Convert to float for BCEWithLogitsLoss
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Squeeze out batch dimension (added by tokenizer with return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.float)
        }

class DistilBertForHumor(nn.Module):
    """DistilBERT-based model for humor detection."""
    
    def __init__(self, model_name, dropout_rate=0.1):
        """
        Args:
            model_name (str): Name of the pretrained DistilBERT model.
            dropout_rate (float): Dropout probability for regularization.
        """
        super(DistilBertForHumor, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Tokenized input sequences.
            attention_mask: Attention mask for padding.
            
        Returns:
            logits: Unnormalized prediction scores.
        """
        # Get DistilBERT outputs
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use the [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0]
        
        # Apply dropout and classification layer
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits.squeeze(-1)  # Remove final dimension for BCEWithLogitsLoss

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train the model for one epoch."""
    model.train()
    epoch_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        
        # Compute loss
        loss = criterion(logits, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update metrics
        epoch_loss += loss.item()
        
        # Convert logits to predictions (binary classification)
        preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    # Calculate average loss
    avg_loss = epoch_loss / len(dataloader)
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics

def evaluate(model, dataloader, criterion, device):
    """Evaluate the model on validation data."""
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            
            # Compute loss
            loss = criterion(logits, labels)
            
            # Update metrics
            val_loss += loss.item()
            
            # Convert logits to predictions (binary classification)
            preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({"val_loss": f"{loss.item():.4f}"})
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    # Calculate average loss
    avg_loss = val_loss / len(dataloader)
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics

def save_checkpoint(model, checkpoint_path):
    """Save model checkpoint."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Save the model state dict
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"Model saved to {checkpoint_path}")

def main(args):
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize tokenizer
    logger.info(f"Loading tokenizer from {args.model_name}")
    tokenizer = DistilBertTokenizer.from_pretrained(args.model_name)
    
    # Create datasets
    train_dataset = HumorTextDataset(
        csv_file=args.train_manifest,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    val_dataset = HumorTextDataset(
        csv_file=args.val_manifest,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Initialize model
    logger.info(f"Loading DistilBERT model from {args.model_name}")
    model = DistilBertForHumor(args.model_name)
    model.to(device)
    
    # Initialize optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs")
    
    # Create log directory if it doesn't exist
    os.makedirs(args.log_dir, exist_ok=True)
    
    # For tracking the best model
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    best_epoch = 0
    
    # For logging metrics across epochs
    training_history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': []
    }
    
    # Training loop
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train one epoch
        train_metrics = train_epoch(model, train_dataloader, optimizer, criterion, device)
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Train Accuracy: {train_metrics['accuracy']:.4f}")
        
        # Evaluate
        val_metrics = evaluate(model, val_dataloader, criterion, device)
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val Accuracy: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1']:.4f}")
        
        # Save metrics for plotting
        training_history['train_loss'].append(train_metrics['loss'])
        training_history['train_accuracy'].append(train_metrics['accuracy'])
        training_history['val_loss'].append(val_metrics['loss'])
        training_history['val_accuracy'].append(val_metrics['accuracy'])
        training_history['val_f1'].append(val_metrics['f1'])
        
        # Check if this is the best model based on validation accuracy
        if val_metrics['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_metrics['accuracy']
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            
            # Save the best model
            save_checkpoint(model, os.path.join('checkpoints', 'text_best.ckpt'))
            logger.info(f"New best model saved (Val Accuracy: {best_val_accuracy:.4f}, Val Loss: {best_val_loss:.4f})")
    
    # Save training history
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_file = os.path.join(args.log_dir, f"training_history_{timestamp}.json")
    
    with open(history_file, 'w') as f:
        json.dump(training_history, f, indent=2)
    logger.info(f"Training history saved to {history_file}")
    
    # Training summary
    logger.info(f"Training completed.")
    logger.info(f"Best model from epoch {best_epoch+1} with Val Accuracy: {best_val_accuracy:.4f}, Val Loss: {best_val_loss:.4f}")
    
    return best_val_accuracy, best_val_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DistilBERT model for humor detection")
    parser.add_argument("--train_manifest", type=str, required=True, help="Path to the training manifest CSV")
    parser.add_argument("--val_manifest", type=str, required=True, help="Path to the validation manifest CSV")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Pretrained model name")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--log_dir", type=str, default="training_logs_text_humor", help="Directory to save training logs")
    
    args = parser.parse_args()
    main(args)
