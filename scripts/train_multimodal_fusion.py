#!/usr/bin/env python
# Train a multimodal fusion model for humor detection
# This script trains a model that combines text, audio, and video modalities

import os
import argparse
import numpy as np
import pandas as pd
import yaml
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import confusion_matrix, classification_report

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging early, before other project imports that might use logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__) # Logger for this script

from dataloaders.fusion_dataset import MultimodalFusionDataset, FusionDataModule
from models.fusion_model import MultimodalFusionModel


def compute_class_weights(manifest_path, split='train'):
    """Compute class weights based on class distribution in training data."""
    df = pd.read_csv(manifest_path)
    train_df = df[df['split'] == split]
    
    # Count labels
    label_counts = train_df['label'].value_counts().sort_index()
    
    # Compute inverse frequency weights
    total = len(train_df)
    weights = total / (len(label_counts) * label_counts)
    
    # Normalize weights to sum to len(label_counts)
    weights = weights * len(label_counts) / weights.sum()
    
    logger.info(f"Computed class weights: {weights.tolist()}")
    return weights.tolist()

def plot_confusion_matrix(y_true, y_pred, classes, output_dir, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    Plot confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap, 
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    logger.info(f"Confusion matrix saved to {os.path.join(output_dir, 'confusion_matrix.png')}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a multimodal fusion model for humor detection')
    
    # Data arguments
    parser.add_argument('--manifest', type=str, required=True,
                        help='Path to the comprehensive multimodal manifest CSV file (talk_id, embedding paths, label, split)')
    # REMOVED: --label_manifest argument as it's no longer needed
    parser.add_argument('--config', type=str, default='configs/model_checkpoint_paths.yaml',
                        help='Path to the model config YAML')
    # REMOVED: --embedding_dir argument as paths in manifest are expected to be absolute or resolvable
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Maximum number of epochs to train for')
    parser.add_argument('--early_stopping', action='store_true',
                        help='Use early stopping')
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience for early stopping')
    parser.add_argument('--class_weights', action='store_true',
                        help='Use class weights for imbalanced data')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    # Model arguments
    parser.add_argument('--fusion_strategy', type=str, default='attention',
                        choices=['early', 'late', 'attention'],
                        help='Fusion strategy to use')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension size')
    parser.add_argument('--output_dim', type=int, default=128,
                        help='Output dimension size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    
    # Modality dimensions (matching defaults in MultimodalFusionModel)
    parser.add_argument('--text_dim', type=int, default=1024,
                        help='Dimension of text embeddings')
    parser.add_argument('--audio_dim', type=int, default=768,
                        help='Dimension of audio embeddings')
    parser.add_argument('--video_dim', type=int, default=1, # Default to 1, will be overridden by command line
                        help='Dimension of video embeddings')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='training_logs_humor',
                        help='Directory to save training logs and models')
    parser.add_argument('--experiment_name', type=str, default='multimodal_fusion',
                        help='Name of the experiment for logging')
    parser.add_argument('--version', type=str, default=None,
                        help='Version of the experiment (optional)')
    
    # Hardware arguments
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--precision', type=int, default=32, choices=[16, 32],
                        help='Training precision')
    
    return parser.parse_args()

def main():
    """Train a multimodal fusion model."""
    
    import sys, importlib, pathlib
    print("\n=== PYTHON SEARCH PATH (sys.path) ===")
    for p in sys.path: print(p)

    try:
        # Attempt the import that might be failing
        import models.fusion_model as fm
        print("\n=== IMPORTED 'models.fusion_model' FROM:", fm.__file__, "===\n")
    except Exception as e:
        print(f"\n=== FAILED TO IMPORT 'models.fusion_model': {e} ===\n")
        # Optionally, try to find where 'models' might be coming from
        try:
            models_spec = importlib.util.find_spec("models")
            if models_spec and models_spec.origin:
                print(f"--- Found 'models' package at: {models_spec.origin}")
            else:
                 print("--- Could not find 'models' package spec.")
        except ModuleNotFoundError:
             print("--- 'models' package not found.")

    args = parse_args()
    
    # Set up experiment directory
    if args.version:
        experiment_dir = os.path.join(args.output_dir, f"{args.experiment_name}_{args.version}")
    else:
        experiment_dir = os.path.join(args.output_dir, args.experiment_name)
    
    os.makedirs(experiment_dir, exist_ok=True)
    logger.info(f"Experiment directory: {experiment_dir}")
    
    # Set up logger
    logger_name = f"{args.experiment_name}"
    if args.version:
        logger_name += f"_{args.version}"
    tb_logger = TensorBoardLogger(args.output_dir, name=logger_name)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set up data module
    from dataloaders.fusion_dataset import FusionDataModule
    # Pass only the main manifest path to FusionDataModule constructor
    data_module = FusionDataModule(
        manifest_path=args.manifest, 
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    data_module.setup()
    
    # Determine embedding dimensions from args (allow override of defaults)
    # Get defaults from the model's argument parser to handle cases where user doesn't provide them
    temp_parser = argparse.ArgumentParser()
    MultimodalFusionModel.add_model_specific_args(temp_parser)
    model_defaults = temp_parser.parse_args([]) # Parse empty list to get defaults

    text_dim = args.text_dim if hasattr(args, 'text_dim') and args.text_dim is not None else model_defaults.text_dim
    audio_dim = args.audio_dim if hasattr(args, 'audio_dim') and args.audio_dim is not None else model_defaults.audio_dim
    video_dim = args.video_dim if hasattr(args, 'video_dim') and args.video_dim is not None else model_defaults.video_dim
    logger.info(f"Using dimensions - Text: {text_dim}, Audio: {audio_dim}, Video: {video_dim}")

    # Compute class weights if requested
    class_weights = None
    if args.class_weights:
        # Class weights should now be computed from the single, comprehensive manifest
        class_weights = compute_class_weights(args.manifest, split='train') 
    
    # Create model based on fusion strategy
    model = MultimodalFusionModel(
        text_dim=text_dim,
        audio_dim=audio_dim,
        video_dim=video_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_classes=2,  # Binary classification for humor
        fusion_strategy=args.fusion_strategy,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        class_weights=class_weights,
        # Pass sequential processing args for audio
        audio_is_sequential=args.audio_is_sequential,
        audio_seq_processor_type=args.audio_seq_processor_type,
        audio_seq_processor_hidden_dim=args.audio_seq_processor_hidden_dim,
        audio_num_seq_processor_layers=args.audio_num_seq_processor_layers,
        # Pass sequential processing args for video
        video_is_sequential=args.video_is_sequential,
        video_seq_processor_type=args.video_seq_processor_type,
        video_seq_processor_hidden_dim=args.video_seq_processor_hidden_dim,
        video_num_seq_processor_layers=args.video_num_seq_processor_layers
    )
    
    # Set up callbacks
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(experiment_dir, 'checkpoints'),
        filename=f'fusion-{args.fusion_strategy}-' + '{epoch:02d}-{val_f1:.3f}',
        monitor='val_f1',
        mode='max',
        save_top_k=3,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    if args.early_stopping:
        early_stopping_callback = EarlyStopping(
            monitor='val_f1',
            mode='max',
            patience=args.patience,
            verbose=True
        )
        callbacks.append(early_stopping_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Configure trainer (PyTorch Lightning v2+ API)
    if torch.cuda.is_available() and args.gpus > 0:
        accelerator = "gpu"
        devices = args.gpus
    else:
        accelerator = "cpu"
        devices = 1

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        precision=args.precision,
        callbacks=callbacks,
        logger=tb_logger,
        log_every_n_steps=50,
        val_check_interval=0.5  # Check validation twice per epoch
    )
    
    # Train model
    logger.info(f"Starting training with fusion strategy: {args.fusion_strategy}")
    trainer.fit(model, data_module)
    
    # Test model
    logger.info("Running final evaluation on test set")
    test_results = trainer.test(model, datamodule=data_module)
    
    # Save test results
    test_results_file = os.path.join(experiment_dir, 'test_results.yaml')
    with open(test_results_file, 'w') as f:
        yaml.dump(test_results, f)
    
    # Generate confusion matrix from validation set
    val_dataloader = data_module.val_dataloader()
    
    # Get all predictions and targets
    all_preds = []
    all_targets = []
    model.eval()
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for batch in val_dataloader:
            text_emb = batch['text_embedding'].to(device)
            audio_emb = batch['audio_embedding'].to(device)
            video_emb = batch['video_embedding'].to(device)
            target = batch['label']
            
            output = model(text_emb, audio_emb, video_emb)
            preds = torch.argmax(output, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(target.cpu().numpy())
    
    # Plot confusion matrix
    plot_confusion_matrix(
        all_targets, all_preds, 
        classes=['Non-Humor', 'Humor'],
        output_dir=experiment_dir
    )
    
    # Print classification report
    logger.info("\nClassification Report:\n")
    report = classification_report(all_targets, all_preds, target_names=['Non-Humor', 'Humor'])
    logger.info(report)
    
    # Save classification report
    with open(os.path.join(experiment_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Save best model path
    logger.info(f"Best model path: {checkpoint_callback.best_model_path}")
    with open(os.path.join(experiment_dir, 'best_model_path.txt'), 'w') as f:
        f.write(checkpoint_callback.best_model_path)
    
    # Export best model to a separate directory for easier deployment
    final_model_dir = os.path.join(experiment_dir, 'final_model')
    os.makedirs(final_model_dir, exist_ok=True)
    
    # Load best model
    best_model = MultimodalFusionModel.load_from_checkpoint(checkpoint_callback.best_model_path)
    
    # Save model state dict
    torch.save(best_model.state_dict(), os.path.join(final_model_dir, 'model.pt'))
    
    # Save model config for later loading (using the determined dimensions)
    model_config = {
        'text_dim': text_dim, # Use determined dim
        'audio_dim': audio_dim, # Use determined dim
        'video_dim': video_dim, # Use determined dim
        'hidden_dim': args.hidden_dim,
        'output_dim': args.output_dim,
        'num_classes': 2,
        'fusion_strategy': args.fusion_strategy,
        'dropout': args.dropout
    }
    
    with open(os.path.join(final_model_dir, 'model_config.yaml'), 'w') as f:
        yaml.dump(model_config, f)
    
    logger.info(f"Training complete. Model saved to {final_model_dir}")
    
    return {
        'best_model_path': checkpoint_callback.best_model_path,
        'final_model_dir': final_model_dir
    }

if __name__ == '__main__':
    main()
