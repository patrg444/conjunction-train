#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Improved XLM-RoBERTa v3 training script with advanced optimization techniques:
- Dynamic batch padding
- Layer-wise learning rate decay
- Label smoothing
- Gradient accumulation
- Linear warmup with cosine decay
- Robust class weighting
- Detailed metrics tracking
- Confusion matrix analysis
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    XLMRobertaTokenizer, 
    XLMRobertaForSequenceClassification, 
    XLMRobertaConfig,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Humor dataset
class HumorDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=128):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['text'])
        label = int(self.data.iloc[idx]['is_humor'])
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Dynamic padding collator for more efficient batching
class DynamicPaddingCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        
    def __call__(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        labels = torch.stack([item['labels'] for item in batch])
        
        # Find max length in the batch
        max_length = max(len(ids) for ids in input_ids)
        
        # Pad all tensors to the max length
        input_ids_padded = []
        attention_mask_padded = []
        
        for ids, mask in zip(input_ids, attention_mask):
            padding_length = max_length - len(ids)
            if padding_length > 0:
                ids = torch.cat([ids, torch.tensor([self.pad_token_id] * padding_length)])
                mask = torch.cat([mask, torch.tensor([0] * padding_length)])
            input_ids_padded.append(ids)
            attention_mask_padded.append(mask)
            
        return {
            'input_ids': torch.stack(input_ids_padded),
            'attention_mask': torch.stack(attention_mask_padded),
            'labels': labels
        }

# Apply layer-wise learning rate decay
def get_optimizer_grouped_parameters(model, args):
    no_decay = ["bias", "LayerNorm.weight"]
    
    # Create layer groups with decay factors
    layers = [model.roberta.embeddings] + list(model.roberta.encoder.layer)
    layers.append(model.classifier)
    
    # Calculate decay factor for each layer
    layer_decay = args.layer_decay
    layer_scales = list(layer_decay ** (len(layers) - i) for i in range(len(layers)))
    
    optimizer_grouped_parameters = []
    for i, layer in enumerate(layers):
        scale = layer_scales[i]
        
        # Parameters with weight decay
        optimizer_grouped_parameters.append(
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
                "lr": args.learning_rate * scale,
            }
        )
        
        # Parameters without weight decay
        optimizer_grouped_parameters.append(
            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": args.learning_rate * scale,
            }
        )
        
    return optimizer_grouped_parameters

# Calculate class weights based on dataset statistics
def calculate_class_weights(train_csv):
    df = pd.read_csv(train_csv)
    class_counts = df['is_humor'].value_counts().to_dict()
    total = len(df)
    weights = {}
    
    for label, count in class_counts.items():
        weights[label] = total / (len(class_counts) * count)
        
    # Convert to tensor of appropriate size
    weight_list = [weights[i] for i in sorted(weights.keys())]
    return torch.tensor(weight_list)

# Label smoothing loss for better generalization
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

# Train function with gradient accumulation
def train_epoch(model, dataloader, optimizer, scheduler, device, criterion, args):
    model.train()
    epoch_loss = 0
    all_preds = []
    all_labels = []
    optimizer.zero_grad()  # Initial zero grad
    
    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        loss = criterion(logits, labels)
        loss = loss / args.accumulation_steps  # Normalize loss for accumulation
        
        loss.backward()
        epoch_loss += loss.item() * args.accumulation_steps  # Scale back for tracking
        
        # Accumulate gradients
        if (step + 1) % args.accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    # Handle any remaining accumulated gradients
    if len(dataloader) % args.accumulation_steps != 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    
    metrics = {
        'loss': epoch_loss / len(dataloader),
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
    
    return metrics

# Validation function
def validate(model, dataloader, device, criterion):
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            loss = criterion(logits, labels)
            epoch_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    
    # Generate and save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    class_names = ['Not Humor', 'Humor']
    
    metrics = {
        'loss': epoch_loss / len(dataloader),
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm.tolist(),
        'class_names': class_names
    }
    
    return metrics, cm, class_names

# Plot and save confusion matrix
def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()

# Save metrics and model checkpoint
def save_metrics(metrics, epoch, log_dir, prefix=""):
    metrics_path = os.path.join(log_dir, f'{prefix}_metrics_epoch_{epoch}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Also append to a log file for easier tracking
    log_path = os.path.join(log_dir, f'{prefix}_training.log')
    metrics_str = f"Epoch {epoch}: " + ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))])
    with open(log_path, 'a') as f:
        f.write(metrics_str + '\n')

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, args, is_best=False):
    checkpoint_dir = os.path.join(args.log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics
    }
    
    if is_best:
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pt'))
    else:
        torch.save(checkpoint, os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pt'))

def main(args):
    # Initialize directories
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Initialize logger to file
    file_handler = logging.FileHandler(os.path.join(args.log_dir, 'training.log'))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    # Log args
    logger.info("Training arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Set seeds for reproducibility
    set_seed(args.seed)
    
    # Initialize tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_version)
    
    # Load datasets
    logger.info(f"Loading datasets from {args.train_path} and {args.val_path}")
    train_dataset = HumorDataset(args.train_path, tokenizer, max_length=args.max_length)
    val_dataset = HumorDataset(args.val_path, tokenizer, max_length=args.max_length)
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Initialize dataloaders with dynamic padding
    collator = DynamicPaddingCollator(tokenizer)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collator,
        num_workers=args.num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collator,
        num_workers=args.num_workers
    )
    
    # Calculate class weights for weighted loss
    class_weights = calculate_class_weights(args.train_path)
    logger.info(f"Class weights: {class_weights}")
    
    # Initialize model
    logger.info(f"Initializing {args.model_version} model")
    config = XLMRobertaConfig.from_pretrained(args.model_version, num_labels=2)
    model = XLMRobertaForSequenceClassification.from_pretrained(args.model_version, config=config)
    
    # Move model and weights to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    model.to(device)
    class_weights = class_weights.to(device)
    
    # Initialize optimizer with layer-wise learning rate decay
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, args)
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Initialize loss function with label smoothing
    if args.label_smoothing > 0:
        criterion = LabelSmoothingLoss(classes=2, smoothing=args.label_smoothing)
        logger.info(f"Using Label Smoothing Loss with smoothing factor {args.label_smoothing}")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        logger.info("Using standard CrossEntropyLoss with class weights")
    
    # Calculate total training steps and warmup steps
    total_steps = len(train_dataloader) * args.num_epochs // args.accumulation_steps
    warmup_steps = int(total_steps * args.warmup_steps_pct)
    logger.info(f"Total training steps: {total_steps}, warmup steps: {warmup_steps}")
    
    # Initialize scheduler with linear warmup and cosine decay
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    logger.info("Starting training...")
    best_f1 = 0.0
    
    for epoch in range(1, args.num_epochs + 1):
        logger.info(f"Epoch {epoch}/{args.num_epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_dataloader, optimizer, scheduler, device, criterion, args)
        logger.info(f"Train metrics: {train_metrics}")
        save_metrics(train_metrics, epoch, args.log_dir, prefix="train")
        
        # Validate
        val_metrics, cm, class_names = validate(model, val_dataloader, device, criterion)
        logger.info(f"Validation metrics: {val_metrics}")
        save_metrics(val_metrics, epoch, args.log_dir, prefix="val")
        
        # Save confusion matrix
        cm_path = os.path.join(args.log_dir, f'confusion_matrix_epoch_{epoch}.png')
        plot_confusion_matrix(cm, class_names, cm_path)
        
        # Save checkpoint
        is_best = val_metrics['f1'] > best_f1
        if is_best:
            best_f1 = val_metrics['f1']
            logger.info(f"New best F1 score: {best_f1:.4f}")
        
        save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, args, is_best)
        
        # Early stopping check
        if args.patience > 0 and epoch > args.patience:
            # Check if validation F1 score has improved in the last 'patience' epochs
            val_metrics_history = []
            for i in range(epoch - args.patience, epoch):
                metrics_path = os.path.join(args.log_dir, f'val_metrics_epoch_{i}.json')
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                    val_metrics_history.append(metrics['f1'])
            
            if len(val_metrics_history) >= args.patience and val_metrics['f1'] <= max(val_metrics_history[:-1]):
                logger.info(f"Early stopping at epoch {epoch} as validation F1 has not improved in the last {args.patience} epochs")
                break
    
    logger.info("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XLM-RoBERTa for humor detection with advanced optimization")
    
    # Data arguments
    parser.add_argument("--train_path", type=str, required=True, help="Path to training CSV file")
    parser.add_argument("--val_path", type=str, required=True, help="Path to validation CSV file")
    
    # Model arguments
    parser.add_argument("--model_version", type=str, default="xlm-roberta-base", 
                      help="XLM-RoBERTa model version (base or large)")
    parser.add_argument("--max_length", type=int, default=128, 
                      help="Maximum sequence length")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--accumulation_steps", type=int, default=4, 
                      help="Number of steps to accumulate gradients")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs and checkpoints")
    
    # Advanced optimization arguments
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--layer_decay", type=float, default=0.95, 
                      help="Layer-wise learning rate decay factor")
    parser.add_argument("--label_smoothing", type=float, default=0.1, 
                      help="Label smoothing factor")
    parser.add_argument("--warmup_steps_pct", type=float, default=0.1, 
                      help="Percentage of warmup steps")
    parser.add_argument("--patience", type=int, default=5, 
                      help="Early stopping patience (set to 0 to disable)")
    
    args = parser.parse_args()
    main(args)
