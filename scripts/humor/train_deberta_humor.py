import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import argparse
from datasets.fixed_text_dataset import TextDataset
import os
import torchmetrics
import pandas as pd
import numpy as np
from torch.nn.utils import clip_grad_norm_

class ImprovedHumorClassifier(pl.LightningModule):
    """
    PyTorch Lightning module for training a Transformer-based humor classifier
    with improved regularization and training recipes.
    """
    def __init__(self, 
                 model_name='microsoft/deberta-v3-base', 
                 num_classes=2, 
                 learning_rate=1e-5, 
                 warmup_steps=100, 
                 total_steps=1000,
                 dropout=0.2,
                 weight_decay=0.01,
                 scheduler_type='cosine',
                 grad_clip=1.0,
                 class_weights=None):
        super().__init__()
        self.save_hyperparameters()

        # Load pre-trained model
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Get hidden dimension from config
        self.hidden_dim = self.bert.config.hidden_size
        
        # Define classifier head with dropout
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_dim, num_classes)
        
        # Use CrossEntropyLoss with optional class weights
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float))
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Use torchmetrics for accuracy and F1
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token embedding for classification
        cls_token_embedding = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(cls_token_embedding)
        logits = self.classifier(pooled_output)
        return logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.criterion(logits, labels)

        # Apply gradient clipping in the optimizer
        if self.hparams.grad_clip > 0:
            clip_grad_norm_(self.parameters(), self.hparams.grad_clip)

        # Calculate metrics
        acc = self.train_accuracy(logits, labels)
        f1 = self.train_f1(logits, labels)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_f1', f1, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.criterion(logits, labels)

        # Calculate metrics
        acc = self.val_accuracy(logits, labels)
        f1 = self.val_f1(logits, labels)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1', f1, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss, 'val_acc': acc, 'val_f1': f1}

    def configure_optimizers(self):
        # Apply weight decay to all parameters except bias and LayerNorm weights
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate)

        # Choose scheduler based on input parameter
        if self.hparams.scheduler_type == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.hparams.total_steps
            )
        elif self.hparams.scheduler_type == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.hparams.total_steps
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {self.hparams.scheduler_type}")
            
        # The scheduler needs to be stepped every training step
        scheduler_config = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler_config]


# Define the collate function needed for DataLoader with dictionary outputs
def collate_dict(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'label': labels}


def compute_class_weights(manifest_path, label_col='label'):
    """Calculate class weights inversely proportional to class frequencies"""
    df = pd.read_csv(manifest_path)
    class_counts = df[label_col].value_counts().sort_index().to_numpy()
    total_samples = np.sum(class_counts)
    n_classes = len(class_counts)
    
    # Compute weights = total_samples / (n_classes * class_counts)
    weights = total_samples / (n_classes * class_counts)
    
    print(f"Class distribution: {class_counts}")
    print(f"Class weights: {weights}")
    
    return weights


def main(args):
    pl.seed_everything(args.seed, workers=True) # for reproducibility

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # --- Data Loaders ---
    train_dataset = TextDataset(manifest_path=args.train_manifest, tokenizer_name=args.model_name, max_length=args.max_length)
    val_dataset = TextDataset(manifest_path=args.val_manifest, tokenizer_name=args.model_name, max_length=args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_dict, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_dict, persistent_workers=args.num_workers > 0)

    # --- Calculate total steps for scheduler ---
    # Estimate total steps: (num_samples / batch_size) * epochs
    total_steps = (len(train_dataset) // args.batch_size) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    print(f"Estimated total training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")

    # --- Calculate class weights if requested ---
    class_weights = None
    if args.class_balanced_loss:
        class_weights = compute_class_weights(args.train_manifest, label_col='label')

    # --- Model ---
    model = ImprovedHumorClassifier(
        model_name=args.model_name,
        num_classes=2,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler,
        grad_clip=args.grad_clip,
        class_weights=class_weights
    )

    # --- Logging & Checkpointing ---
    log_dir = os.path.join(args.log_dir, args.exp_name)
    logger = TensorBoardLogger(save_dir=log_dir, name="lightning_logs")

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_dir, "checkpoints"),
        filename='{epoch:02d}-{val_f1:.4f}', # Save based on validation F1
        save_top_k=1,
        verbose=True,
        monitor='val_f1', # Monitor validation F1 score
        mode='max'
    )
    lr_monitor = LearningRateMonitor(logging_interval='step') # Log LR every step
    early_stopping_callback = EarlyStopping(
        monitor='val_f1', # Stop if validation F1 doesn't improve
        patience=args.early_stop_patience,
        verbose=True,
        mode='max'
    )

    # --- Trainer ---
    # Use deterministic=True if reproducibility is critical
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stopping_callback],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=args.gpus,
        precision=16 if args.fp16 else 32, # Mixed precision
        deterministic=False, # Set False for speed
        log_every_n_steps=20 # Log more frequently
    )

    # --- Training ---
    print(f"Starting {args.model_name} humor classification training for {args.epochs} epochs...")
    trainer.fit(model, train_loader, val_loader)
    print("Training finished.")
    
    # Print best score for easier logging
    best_checkpoint = checkpoint_callback.best_model_path
    best_score = checkpoint_callback.best_model_score
    print(f"\nBest model checkpoint: {best_checkpoint}")
    print(f"Best validation F1 score: {best_score:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Transformer-based Humor Classifier')

    # Data args
    parser.add_argument('--train_manifest', type=str, required=True, help='Path to the training manifest CSV')
    parser.add_argument('--val_manifest', type=str, required=True, help='Path to the validation manifest CSV')
    parser.add_argument('--model_name', type=str, default='microsoft/deberta-v3-base', 
                      help='Name of the Hugging Face transformer model. Options: microsoft/deberta-v3-base, microsoft/deberta-v3-small, roberta-base')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length for tokenizer')

    # Training args
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--epochs', type=int, default=3, help='Maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate for the classifier layer')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for AdamW optimizer')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Ratio of steps for warmup (0-1)')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['linear', 'cosine'], help='Learning rate scheduler type')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value (0 to disable)')
    parser.add_argument('--early_stop_patience', type=int, default=2, help='Patience for early stopping')
    parser.add_argument('--class_balanced_loss', action='store_true', help='Use class-balanced loss weighting')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use (0 for CPU)')
    parser.add_argument('--fp16', action='store_true', help='Enable mixed precision training')

    # Logging args
    parser.add_argument('--log_dir', type=str, default='training_logs_humor', help='Directory to save logs and checkpoints')
    parser.add_argument('--exp_name', type=str, default='deberta_humor', help='Experiment name for logging')

    args = parser.parse_args()
    main(args)
