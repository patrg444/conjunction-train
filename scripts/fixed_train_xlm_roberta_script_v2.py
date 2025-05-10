import os
import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
# Fix: Import AdamW from torch.optim instead of transformers
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim import AdamW  # Changed import to use PyTorch's AdamW
from torch.utils.data import Dataset, DataLoader
import argparse
from typing import Optional, Dict, List
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
from collections import Counter  # Add this import for class balancing

class HumorDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=128):
        self.df = pd.read_csv(csv_file)
        # Handle different column names between datasets
        if 'text' in self.df.columns:
            self.text_column = 'text'
            self.label_column = 'is_humor'
        elif 'transcript' in self.df.columns:
            self.text_column = 'transcript'
            self.label_column = 'label'
        else:
            raise ValueError("CSV must contain either 'text' or 'transcript' column")
            
        self.texts = self.df[self.text_column].tolist()
        self.labels = self.df[self.label_column].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Don't add padding here - will be done in collate_fn for efficiency
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def collate_fn(batch):
    """
    Custom collate function for dynamic padding to the longest sequence in batch
    """
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Pad input_ids and attention_masks to the max length in the batch
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True)
    attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels = torch.stack(labels)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'labels': labels
    }

# Define label smoothing loss
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

class XLMRobertaHumorClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        num_classes: int = 2,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        weight_decay: float = 0.01,
        scheduler_type: str = "cosine",
        warmup_steps: int = 0,
        total_steps: int = 0,
        dropout: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.1,
        layer_decay: float = 0.95
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = XLMRobertaForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout
        )

        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        self.layer_decay = layer_decay
        
        # Initialize label smoothing loss
        if self.label_smoothing > 0:
            self.criterion = LabelSmoothingLoss(classes=num_classes, smoothing=label_smoothing)
        else:
            self.criterion = None  # Will use default loss or weighted cross entropy

        # Store outputs for validation epoch end processing
        self.validation_step_outputs = []
        
        # Save confusion matrices for visualization
        self.confusion_matrices = []

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )

        logits = outputs.logits
        labels = batch['labels']

        # Apply label smoothing or class weights to loss
        if self.criterion is not None:
            loss = self.criterion(logits, labels)
        elif self.class_weights is not None:
            # Use F.cross_entropy with weights instead of default loss
            loss = F.cross_entropy(
                logits,
                labels,
                weight=self.class_weights.to(labels.device)
            )
        else:
            loss = outputs.loss

        preds = torch.argmax(logits, dim=1)

        # Calculate accuracy
        acc = (preds == labels).float().mean()

        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )

        loss = outputs.loss
        logits = outputs.logits

        preds = torch.argmax(logits, dim=1)
        labels = batch['labels']

        # Store outputs for epoch end processing
        self.validation_step_outputs.append({
            'loss': loss,
            'preds': preds,
            'labels': labels
        })

        return {'loss': loss, 'preds': preds, 'labels': labels}

    def on_validation_epoch_end(self):
        # Process all outputs from the validation steps
        outputs = self.validation_step_outputs

        # Gather all predictions and labels
        all_preds = torch.cat([x['preds'] for x in outputs])
        all_labels = torch.cat([x['labels'] for x in outputs])

        # Calculate accuracy
        acc = (all_preds == all_labels).float().mean()

        # Convert tensors to numpy for precision, recall, f1
        preds_np = all_preds.cpu().numpy()
        labels_np = all_labels.cpu().numpy()

        # Calculate precision, recall, f1
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_np, preds_np, average='weighted', zero_division=0
        )

        # Calculate and log confusion matrix
        cm = confusion_matrix(labels_np, preds_np)
        self.confusion_matrices.append(cm)
        
        # Save confusion matrix to file in the log directory if available
        if self.logger and hasattr(self.logger, 'log_dir'):
            log_dir = self.logger.log_dir
            epoch = self.current_epoch
            
            # Create confusion matrix plot
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Not Humor', 'Humor'],
                       yticklabels=['Not Humor', 'Humor'])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix - Epoch {epoch}')
            
            # Create directory if it doesn't exist
            os.makedirs(log_dir, exist_ok=True)
            plt.savefig(os.path.join(log_dir, f'confusion_matrix_epoch_{epoch}.png'))
            plt.close()

        # Calculate confusion matrix details (for binary case)
        if self.hparams.num_classes == 2:
            tn = ((preds_np == 0) & (labels_np == 0)).sum()
            fp = ((preds_np == 1) & (labels_np == 0)).sum()
            fn = ((preds_np == 0) & (labels_np == 1)).sum()
            tp = ((preds_np == 1) & (labels_np == 1)).sum()

            # Log confusion matrix elements
            self.log('val_tp', tp, on_epoch=True, prog_bar=False)
            self.log('val_fp', fp, on_epoch=True, prog_bar=False)
            self.log('val_tn', tn, on_epoch=True, prog_bar=False)
            self.log('val_fn', fn, on_epoch=True, prog_bar=False)

            # Calculate class-wise metrics
            precision_class1 = tp / (tp + fp) if tp + fp > 0 else 0
            recall_class1 = tp / (tp + fn) if tp + fn > 0 else 0
            f1_class1 = 2 * precision_class1 * recall_class1 / (precision_class1 + recall_class1) if precision_class1 + recall_class1 > 0 else 0

            self.log('val_precision_class1', precision_class1, on_epoch=True, prog_bar=False)
            self.log('val_recall_class1', recall_class1, on_epoch=True, prog_bar=False)
            self.log('val_f1_class1', f1_class1, on_epoch=True, prog_bar=False)

        # Average validation loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # Log validation metrics
        self.log('val_loss', avg_loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        self.log('val_precision', precision, on_epoch=True, prog_bar=True)
        self.log('val_recall', recall, on_epoch=True, prog_bar=True)
        self.log('val_f1', f1, on_epoch=True, prog_bar=True)
        
        # Save metrics to JSON for easier analysis
        if self.logger and hasattr(self.logger, 'log_dir'):
            log_dir = self.logger.log_dir
            epoch = self.current_epoch
            metrics = {
                'epoch': epoch,
                'val_loss': avg_loss.item(),
                'val_acc': acc.item(),
                'val_precision': precision,
                'val_recall': recall,
                'val_f1': f1,
                'confusion_matrix': cm.tolist()
            }
            
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, f'metrics_epoch_{epoch}.json'), 'w') as f:
                json.dump(metrics, f, indent=2)

        # Clear the list for the next epoch
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        # Apply layer-wise learning rate decay
        if self.layer_decay < 1.0:
            # Group parameters by layer for different learning rates
            no_decay = ["bias", "LayerNorm.weight"]
            
            # Create layer groups with decay factors
            layers = [self.model.roberta.embeddings] + list(self.model.roberta.encoder.layer)
            layers.append(self.model.classifier)  # Add classifier as final layer
            
            # Calculate decay factor for each layer (lower LR for lower layers)
            layer_scales = list(self.layer_decay ** (len(layers) - i) for i in range(len(layers)))
            
            optimizer_grouped_parameters = []
            for i, layer in enumerate(layers):
                scale = layer_scales[i]
                
                # Parameters with weight decay
                optimizer_grouped_parameters.append({
                    "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.weight_decay,
                    "lr": self.learning_rate * scale,
                })
                
                # Parameters without weight decay
                optimizer_grouped_parameters.append({
                    "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "lr": self.learning_rate * scale,
                })
                
            optimizer = AdamW(optimizer_grouped_parameters, eps=self.adam_epsilon)
        else:
            # Standard optimizer without layer-wise decay
            optimizer = AdamW(
                self.parameters(),
                lr=self.learning_rate,
                eps=self.adam_epsilon,
                weight_decay=self.weight_decay
            )

        # Get total steps from the trainer if not explicitly set
        # This fixes the multi-GPU or gradient accumulation issue
        if self.trainer and hasattr(self.trainer, 'estimated_stepping_batches'):
            # New Lightning API (2.0+)
            total_steps = self.trainer.estimated_stepping_batches
        elif self.trainer:
            # Fallback for older Lightning versions
            if hasattr(self.trainer, 'max_steps') and self.trainer.max_steps > 0:
                total_steps = self.trainer.max_steps
            else:
                total_steps = len(self.trainer.datamodule.train_dataloader()) * self.trainer.max_epochs
                if hasattr(self.trainer, 'accumulate_grad_batches'):
                    total_steps = total_steps // self.trainer.accumulate_grad_batches
                if hasattr(self.trainer, 'num_devices') and self.trainer.num_devices > 1:
                    total_steps = total_steps // self.trainer.num_devices
        else:
            total_steps = self.total_steps

        # Recompute warmup if total steps changed
        warmup_steps = int(0.1 * total_steps)  # 10% warmup

        # Define scheduler
        if self.scheduler_type == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        elif self.scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

def parse_args():
    parser = argparse.ArgumentParser(description='Train XLM-RoBERTa for humor detection (V3 Advanced Optimization)')

    parser.add_argument('--train_manifest', type=str, required=True, help='Path to training manifest CSV')
    parser.add_argument('--val_manifest', type=str, required=True, help='Path to validation manifest CSV')
    parser.add_argument('--model_name', type=str, default='xlm-roberta-large', help='HuggingFace model name')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['linear', 'cosine'], help='Scheduler type')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--log_dir', type=str, default='training_logs_humor', help='Log directory')
    parser.add_argument('--exp_name', type=str, default='xlm-roberta-large_v3_optimized', help='Experiment name')
    parser.add_argument('--devices', type=int, default=1, help='Number of devices to use')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision training')
    parser.add_argument('--class_balancing', action='store_true', help='Use class weights to balance training')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor (0 to disable)')
    parser.add_argument('--layer_decay', type=float, default=0.95, help='Layer-wise learning rate decay factor')
    parser.add_argument('--warmup_steps_pct', type=float, default=0.1, help='Percentage of steps for warmup')
    parser.add_argument('--accumulation_steps', type=int, default=4, help='Number of steps to accumulate gradients')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--monitor_metric', type=str, default='val_f1', choices=['val_loss', 'val_acc', 'val_f1'],
                        help='Metric to monitor for early stopping and checkpointing')

    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    pl.seed_everything(args.seed)

    print(f"Starting V3 advanced training with {args.epochs} epochs at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {args.model_name}")
    print(f"Training manifest: {args.train_manifest}")
    print(f"Validation manifest: {args.val_manifest}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Using class balancing: {args.class_balancing}")
    print(f"Label smoothing: {args.label_smoothing}")
    print(f"Layer decay: {args.layer_decay}")
    print(f"Gradient accumulation steps: {args.accumulation_steps}")
    print(f"Monitoring metric: {args.monitor_metric}")

    # Set up model name for logging
    model_name_safe = args.model_name.replace('/', '_')

    # Set up tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_name)

    # Load datasets
    train_dataset = HumorDataset(
        csv_file=args.train_manifest,
        tokenizer=tokenizer,
        max_length=args.max_length
    )

    val_dataset = HumorDataset(
        csv_file=args.val_manifest,
        tokenizer=tokenizer,
        max_length=args.max_length
    )

    print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    print(f"Text column: {train_dataset.text_column}, Label column: {train_dataset.label_column}")

    # Check class distribution for potential imbalance
    class_weights = None
    if args.class_balancing:
        labels = [item for item in train_dataset.labels]
        label_counts = Counter(labels)
        print(f"Class distribution in training set: {label_counts}")
        
        # Calculate class weights inversely proportional to class frequencies
        total_samples = len(labels)
        weights = {}
        for label, count in label_counts.items():
            weights[label] = total_samples / (len(label_counts) * count)
        
        class_weights = torch.tensor([weights[i] for i in range(len(label_counts))], dtype=torch.float)
        print(f"Using class weights: {class_weights}")

    # Create data loaders with custom collate_fn for dynamic padding
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0
    )

    # Calculate steps for scheduler
    total_steps = len(train_loader) * args.epochs // args.accumulation_steps
    warmup_steps = int(args.warmup_steps_pct * total_steps)

    print(f"Estimated training steps: {total_steps}, with {warmup_steps} warmup steps")

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.log_dir, args.exp_name, 'checkpoints'),
        filename='{epoch}-{val_loss:.2f}-{' + args.monitor_metric + ':.4f}',
        save_top_k=3,
        monitor=args.monitor_metric,
        mode='max' if args.monitor_metric != 'val_loss' else 'min'
    )

    early_stop_callback = EarlyStopping(
        monitor=args.monitor_metric,
        patience=5,  # Patience of 5 epochs
        mode='max' if args.monitor_metric != 'val_loss' else 'min'
    )

    # Set up logger
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.exp_name
    )

    # Create model
    model = XLMRobertaHumorClassifier(
        model_name=args.model_name,
        num_classes=2,  # Binary classification (0: not humor, 1: humor)
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        dropout=args.dropout,
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
        layer_decay=args.layer_decay
    )

    # Create trainer with updated Lightning API
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        max_epochs=args.epochs,
        devices=args.devices,
        precision="16-mixed" if args.fp16 else 32,
        gradient_clip_val=args.grad_clip,
        log_every_n_steps=10,
        accelerator="auto",  # Let Lightning auto-detect the best accelerator
        deterministic=True,  # For reproducibility
        accumulate_grad_batches=args.accumulation_steps  # Accumulate gradients for effective larger batch size
    )

    # Train model
    print(f"Starting training with {args.epochs} epochs")
    trainer.fit(model, train_loader, val_loader)

    # Save the final model
    model_path = os.path.join(args.log_dir, args.exp_name, 'final_model')
    model.model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    print(f"Training completed at {time.strftime('%Y-%m-%d %H:%M:%S')}! Model saved to {model_path}")

if __name__ == '__main__':
    main()
