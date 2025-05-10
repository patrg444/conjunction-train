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
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time

class HumorDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=128):
        self.df = pd.read_csv(csv_file)
        # Fix: Use 'transcript' column instead of 'text'
        self.texts = self.df['transcript'].tolist()
        self.labels = self.df['label'].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class XLMRobertaHumorClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        num_classes: int = 2,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        weight_decay: float = 0.01,
        scheduler_type: str = "linear",
        warmup_steps: int = 0,
        total_steps: int = 0,
        dropout: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
        grad_clip: float = 1.0
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
        self.grad_clip = grad_clip

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

        loss = outputs.loss
        logits = outputs.logits

        preds = torch.argmax(logits, dim=1)
        labels = batch['labels']

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

        # Return predictions and labels for epoch end validation metrics
        return {'loss': loss, 'preds': preds, 'labels': labels}

    def validation_epoch_end(self, outputs):
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

        # Calculate confusion matrix (for binary case)
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

    def configure_optimizers(self):
        # Define optimizer (now using torch.optim.AdamW)
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            eps=self.adam_epsilon,
            weight_decay=self.weight_decay
        )

        # Define scheduler
        if self.scheduler_type == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.total_steps
            )
        elif self.scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.total_steps
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

def parse_args():
    parser = argparse.ArgumentParser(description='Train XLM-RoBERTa for humor detection (Extended Training)')

    parser.add_argument('--train_manifest', type=str, required=True, help='Path to training manifest CSV')
    parser.add_argument('--val_manifest', type=str, required=True, help='Path to validation manifest CSV')
    parser.add_argument('--model_name', type=str, default='xlm-roberta-large', help='HuggingFace model name')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['linear', 'cosine'], help='Scheduler type')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--log_dir', type=str, default='training_logs_humor', help='Log directory')
    parser.add_argument('--exp_name', type=str, default='xlm-roberta-large_extended_training', help='Experiment name')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision training')

    return parser.parse_args()

def main():
    args = parse_args()

    print(f"Starting extended training with {args.epochs} epochs at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {args.model_name}")
    print(f"Training manifest: {args.train_manifest}")
    print(f"Validation manifest: {args.val_manifest}")

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

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # Calculate steps for scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)  # 10% warmup

    print(f"Total training steps: {total_steps}, with {warmup_steps} warmup steps")

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.log_dir, args.exp_name, 'checkpoints'),
        filename='{epoch}-{val_loss:.2f}-{val_acc:.4f}',
        save_top_k=3,
        monitor='val_acc',
        mode='max'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,  # Increased patience for longer training
        mode='min'
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
        grad_clip=args.grad_clip
    )

    # Create trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        max_epochs=args.epochs,
        gpus=args.gpus,
        precision=16 if args.fp16 else 32,
        gradient_clip_val=args.grad_clip,
        log_every_n_steps=10,
        accelerator="gpu" if args.gpus > 0 else None,
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
