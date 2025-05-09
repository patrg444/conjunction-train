import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import yaml
import math
from pathlib import Path
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from transformers import get_cosine_schedule_with_warmup

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, balanced_accuracy_score
from datetime import datetime

# Add project root to path to allow sibling imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added project root to sys.path: {project_root}")

# Import the custom dataset and model
try:
    from datasets.urfunny_audio_dataset import URFunnyAudioDataset
    print("Successfully imported URFunnyAudioDataset")
except ImportError as e:
    print(f"Error importing URFunnyAudioDataset: {e}")
    sys.exit(1)

try:
    from models.wavlm_classifier import WavLMClassifier
    print("Successfully imported WavLMClassifier")
except ImportError as e:
    print(f"Error importing WavLMClassifier: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---
def load_yaml(file_path):
    """Load a YAML configuration file."""
    try:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"Error loading YAML file {file_path}: {e}")
        return None

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

# --- Training and Validation Functions ---
def train_epoch(model, dataloader, criterion, optimizer, scheduler, scaler, device, use_amp):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        if batch is None:
            continue

        input_values = batch['input_values'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].long().to(device)

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            logits = model(input_values=input_values, attention_mask=attention_mask)
            loss = criterion(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()

        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0) * 100

    return epoch_loss, accuracy, f1

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    progress_bar = tqdm(dataloader, desc="Validating")

    with torch.no_grad():
        for batch in progress_bar:
            if batch is None:
                continue
            input_values = batch['input_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].long().to(device)

            logits = model(input_values=input_values, attention_mask=attention_mask)
            loss = criterion(logits, labels)

            running_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds) * 100
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    balanced_acc = balanced_accuracy_score(all_labels, all_preds) * 100

    print("\n--- Validation Results ---")
    print(f"Accuracy: {accuracy:.2f}%, F1 Score (Binary): {f1_score(all_labels, all_preds, average='binary', zero_division=0) * 100:.2f}%, F1 Score (Macro): {macro_f1:.2f}%, Balanced Accuracy: {balanced_acc:.2f}%")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Non-Humor', 'Humor'], zero_division=0))

    # Optional: Print confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)

    return val_loss, accuracy, macro_f1

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Train Audio-Only WavLM Humor Classifier")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file (e.g., configs/train_urfunny_audio.yaml)")
    parser.add_argument("--freeze_backbone_epochs", type=int, default=0, help="Number of epochs to freeze the backbone.")

    args = parser.parse_args()

    # --- Load Configuration ---
    if not os.path.exists(args.config):
        logging.error(f"Config file not found at {args.config}")
        sys.exit(1)
    config = load_yaml(args.config)
    if config is None:
        sys.exit(1)

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Set cudnn benchmark for potential speedup
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logging.info("torch.backends.cudnn.benchmark set to True.")

    output_dir = config['training']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Checkpoints and logs will be saved to: {output_dir}")

    # --- Data Loading ---
    logging.info("Creating Datasets and DataLoaders...")
    dataset_params = config['data']
    train_manifest_path = dataset_params['train_manifest']
    val_manifest_path = dataset_params.get('val_manifest', None)

    if not os.path.exists(train_manifest_path):
        logging.error(f"Train manifest not found at {train_manifest_path}")
        sys.exit(1)

    train_dataset = URFunnyAudioDataset(
        manifest_path=train_manifest_path,
        sample_rate=dataset_params.get('sample_rate', 16000),
        duration=dataset_params.get('duration', 3.0), # Use duration from config
        model_name=config['model']['model_name'],
        data_root=dataset_params.get('data_root') # Pass data_root from config
    )

    val_dataset = None
    if val_manifest_path and os.path.exists(val_manifest_path):
        val_dataset = URFunnyAudioDataset(
            manifest_path=val_manifest_path,
            sample_rate=dataset_params.get('sample_rate', 16000),
            duration=dataset_params.get('duration', 3.0), # Use duration from config
            model_name=config['model']['model_name'],
            data_root=dataset_params.get('data_root') # Pass data_root from config
        )
        logging.info(f"Validation manifest found: {val_manifest_path}")
    else:
        logging.warning("No validation manifest provided or found. Validation will be skipped.")

    dataloader_params = config['data'].get('dataloader_params', {})

    logging.info("Creating DataLoaders...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=dataloader_params.get('batch_size', 16),
        shuffle=True,
        num_workers=dataloader_params.get('num_workers', 4),
        pin_memory=dataloader_params.get('pin_memory', True),
        collate_fn=safe_collate_fn
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=dataloader_params.get('batch_size', 16),
            shuffle=False,
            num_workers=dataloader_params.get('num_workers', 4),
            pin_memory=dataloader_params.get('pin_memory', True),
            collate_fn=safe_collate_fn
        )

    logging.info("DataLoaders created.")

    # --- Model Setup ---
    logging.info("Creating WavLM Classifier Model...")
    model_config = config['model']
    model = WavLMClassifier(
        model_name=model_config.get('model_name', "microsoft/wavlm-base-plus"),
        num_classes=model_config.get('num_classes', 2),
        freeze_feature_extractor=model_config.get('freeze_feature_extractor', False)
    ).to(device)
    logging.info("Model created.")

    # --- Optimizer, Criterion, Scaler ---
    training_config = config['training']
    param_groups = [
        {'params': model.wavlm_classifier.classifier.parameters(), 'lr': training_config['optimizer_params']['classifier_lr']},
        {'params': model.wavlm_classifier.projector.parameters(), 'lr': training_config['optimizer_params']['classifier_lr']},
        {'params': model.wavlm_classifier.wavlm.parameters(), 'lr': training_config['optimizer_params']['backbone_lr']}
    ]

    optimizer = optim.AdamW(param_groups,
                            weight_decay=training_config['optimizer_params']['weight_decay'])

    # Bias initialisation (log-prior trick)
    try:
        train_labels = pd.read_csv(train_manifest_path)['label'].values
        class_counts = np.bincount(train_labels)
        if len(class_counts) < 2:
            if 0 not in train_labels:
                 class_counts = np.array([class_counts[0], 0])
            else:
                 class_counts = np.array([0, class_counts[0]])

        pos = class_counts[1] if len(class_counts) > 1 else 0
        neg = class_counts[0] if len(class_counts) > 0 else 0
        if (pos + neg) > 0:
            p = pos / (pos + neg)
            if p == 0:
                bias = -10.0
            elif p == 1:
                bias = 10.0
            else:
                bias = math.log(p / (1 - p))
            with torch.no_grad():
                model.wavlm_classifier.classifier.bias.fill_(bias)
            logging.info(f"Bias initialized with value: {bias:.4f}")
        else:
            logging.warning("Total samples is 0, skipping bias initialization.")
    except Exception as e:
        logging.warning(f"Could not perform bias initialization: {e}")

    criterion = nn.CrossEntropyLoss()
    logging.info("Using CrossEntropyLoss.")

    use_amp = training_config.get('fp16', True)
    scaler = torch.amp.GradScaler(enabled=use_amp) if use_amp else None
    logging.info(f"Using Mixed Precision (AMP): {use_amp}")

    # --- LR Scheduler Setup ---
    scheduler = None
    lr_scheduler_config = training_config.get('lr_scheduler', None)

    if lr_scheduler_config is not None and lr_scheduler_config.get('name') == 'cosine':
        num_training_steps = len(train_loader) * training_config.get('epochs', 20)
        num_warmup_steps = int(num_training_steps * lr_scheduler_config.get('warmup_pct', 0.0))

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        logging.info(f"Using Cosine LR scheduler with {num_warmup_steps} warmup steps and {num_training_steps} total steps.")
    elif lr_scheduler_config is not None:
        logging.warning(f"Unsupported LR scheduler '{lr_scheduler_config.get('name')}'. No scheduler will be used.")
    else:
        logging.info("No LR scheduler configured.")

    # --- Training Loop ---
    epochs = training_config.get('epochs', 20)
    early_stop_patience = training_config.get('early_stop_patience', 5)
    best_val_f1 = 0.0
    patience_counter = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_name = f"wavlm_urfunny_{timestamp}"

    logging.info("Starting training loop...")
    for epoch in range(1, epochs + 1):
        logging.info(f"\n--- Epoch {epoch}/{epochs} ---")
        for i, param_group in enumerate(optimizer.param_groups):
             logging.info(f"Current LR (Group {i}): {param_group['lr']:.2e}")

        # Freeze/Unfreeze backbone based on args.freeze_backbone_epochs
        if args.freeze_backbone_epochs > 0:
            if epoch <= args.freeze_backbone_epochs:
                logging.info(f"Freezing backbone for epoch {epoch}.")
                for param in model.wavlm_classifier.wavlm.parameters():
                    param.requires_grad = False
            else:
                if epoch == args.freeze_backbone_epochs + 1:
                    logging.info(f"Unfreezing backbone for epoch {epoch}.")
                    for param in model.wavlm_classifier.wavlm.parameters():
                        param.requires_grad = True

        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device, use_amp
        )
        logging.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Train F1={train_f1:.2f}%")

        if val_loader:
            val_loss, val_acc, val_f1 = validate(
                model, val_loader, criterion, device
            )
            logging.info(f"Epoch {epoch}: Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, Val F1={val_f1:.2f}%")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                logging.info(f"Saving best model with Validation F1: {best_val_f1:.2f}%")
                save_path = os.path.join(output_dir, f"{model_save_name}_best.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_f1': best_val_f1,
                    'config': config
                }, save_path)
            else:
                patience_counter += 1
                logging.info(f"Validation F1 did not improve. Patience: {patience_counter}/{early_stop_patience}")
                if patience_counter >= early_stop_patience:
                    logging.info(f"Early stopping triggered after {early_stop_patience} epochs without improvement.")
                    break
        else:
            logging.info("No validation loader, skipping checkpointing based on validation metric.")

    logging.info(f"\nTraining finished. Best Validation F1: {best_val_f1:.2f}%")
    if val_loader:
        logging.info(f"Best model saved to: {os.path.join(output_dir, f'{model_save_name}_best.pt')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Audio-Only WavLM Humor Classifier")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file (e.g., configs/train_urfunny_audio.yaml)")
    parser.add_argument("--verify", action="store_true", help="Run verification steps instead of training.")
    parser.add_argument("--freeze_backbone_epochs", type=int, default=0, help="Number of epochs to freeze the backbone.")

    args = parser.parse_args()

    if args.verify:
        print("--- Running Verification Steps ---")
        config_path = args.config
        if not os.path.exists(config_path):
            print(f"Error: Config file not found at {config_path}")
            sys.exit(1)
        config = load_yaml(config_path)
        if config is None:
            sys.exit(1)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        dataset_params = config['data']
        train_manifest_path = dataset_params['train_manifest']
        val_manifest_path = dataset_params.get('val_manifest', None) # Get val_manifest_path here

        if not os.path.exists(train_manifest_path):
             print(f"Error: Train manifest not found at {train_manifest_path}")
             sys.exit(1)

        train_dataset = URFunnyAudioDataset(
            manifest_path=train_manifest_path,
            sample_rate=dataset_params.get('sample_rate', 16000),
            duration=dataset_params.get('duration', 3.0), # Use duration from config
            model_name=config['model']['model_name'],
            data_root=dataset_params.get('data_root') # Pass data_root from config
        )

        # Create train_loader here for verification
        dataloader_params = config['data'].get('dataloader_params', {})
        train_loader = DataLoader(
            train_dataset,
            batch_size=dataloader_params.get('batch_size', 16),
            shuffle=False, # No need to shuffle for this test
            num_workers=dataloader_params.get('num_workers', 4),
            pin_memory=dataloader_params.get('pin_memory', True),
            collate_fn=safe_collate_fn # Use the safe collate function
        )


        model_config = config['model']
        model = WavLMClassifier(
            model_name=model_config.get('model_name', 'microsoft/wavlm-base-plus'),
            num_classes=model_config.get('num_classes', 2),
            freeze_feature_extractor=model_config.get('freeze_feature_extractor', False)
        ).to(device)

        try:
            b = next(iter(train_loader))
        except StopIteration:
            print("Error: Training data loader is empty.")
            sys.exit(1)
        except Exception as e:
            print(f"Error getting first batch: {e}")
            sys.exit(1)

        with torch.no_grad():
            logits = model(b['input_values'].to(device),
                           attention_mask=b['attention_mask'].to(device))
        print('Logits shape:', logits.shape)

        print('First batch label counts:', np.unique(b['label'].cpu().numpy(), return_counts=True))

        print("--- Verification Steps Finished ---")

    else:
        main()
