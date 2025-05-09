#!/usr/bin/env python3
"""
Train a 3D-ResNet-LSTM model on combined RAVDESS and CREMA-D datasets.
Uses video-only input to predict emotion labels.

Key components:
- 3D ResNet-18 backbone
- Global average pooling
- LSTM layer
- Fully connected layer
- Mixed precision training
- Cosine LR schedule
"""

import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import torchvision
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime

# Emotion labels that are common between RAVDESS and CREMA-D
EMOTION_LABELS = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad']


class VideoDataset(Dataset):
    """Dataset loader for video emotional classification."""
    
    def __init__(self, manifest_file, split='train', frames=48, img_size=112, augment=True):
        """
        Initialize the dataset.
        
        Args:
            manifest_file: CSV file containing video filepaths and labels
            split: 'train', 'val', or 'test'
            frames: Number of frames to sample from each video
            img_size: Size to resize video frames to (square)
            augment: Whether to use data augmentation for training
        """
        self.manifest = pd.read_csv(manifest_file)
        self.manifest = self.manifest[self.manifest['split'] == split]
        self.label_to_idx = {label: i for i, label in enumerate(EMOTION_LABELS)}
        self.frames = frames
        self.img_size = img_size
        self.augment = augment and split == 'train'
        
        # Keep only the emotions that are in EMOTION_LABELS
        self.manifest = self.manifest[self.manifest['label'].isin(EMOTION_LABELS)]
        self.manifest = self.manifest.reset_index(drop=True)

        print(f"Loaded {len(self.manifest)} videos for split '{split}'")
        print(self.manifest['label'].value_counts())
        
        # Setup augmentation and normalization transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.manifest)
    
    def __getitem__(self, index):
        """Get video item by index."""
        video_path = self.manifest.iloc[index]['filepath']
        emotion = self.manifest.iloc[index]['label']
        label_idx = self.label_to_idx[emotion]
        
        # Load video frames
        frames = self._load_video(video_path)
        
        # Convert to tensor and normalize
        frames = torch.stack([self.transform(frame) for frame in frames])
        
        return frames, label_idx
    
    def _load_video(self, video_path):
        """Load video frames from file."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # If there aren't enough frames, loop the video
        if frame_count < self.frames:
            # Read all available frames
            while len(frames) < frame_count:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            # Handle case where no frames were successfully read
            if not frames:
                # Create a blank frame if no frames were read
                blank_frame = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                frames = [blank_frame] * self.frames
            else:
                # Loop frames until we reach required count
                while len(frames) < self.frames:
                    frames.append(frames[len(frames) % len(frames)])
        else:
            # Sample frames evenly from the video
            indices = np.linspace(0, frame_count - 1, self.frames, dtype=int)
            
            for i in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    # If reading fails, use the last valid frame
                    if frames:
                        frames.append(frames[-1])
                    else:
                        # If no valid frames, create a black frame
                        frames.append(np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8))
                    continue
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        
        # Resize frames
        frames = [cv2.resize(frame, (self.img_size, self.img_size)) for frame in frames]
        
        # Apply augmentations for training
        if self.augment:
            # Random horizontal flip (same for all frames)
            if random.random() > 0.5:
                frames = [np.fliplr(frame) for frame in frames]
            
            # Random brightness and contrast adjustment
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            frames = [np.clip(frame * brightness + contrast - brightness * contrast * 127.5 + 127.5 * (1 - contrast), 0, 255).astype(np.uint8) for frame in frames]
        
        return frames


class VideoEmbedder(nn.Module):
    """3D CNN for embedding video frames."""
    
    def __init__(self, pretrained=True, freeze_backbone=True):
        """Initialize the video embedder."""
        super(VideoEmbedder, self).__init__()
        
        # Use ResNet-18 3D as the backbone
        self.backbone = torchvision.models.video.r3d_18(pretrained=pretrained)
        
        # Freeze backbone weights
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Remove the final fully connected layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
    
    def forward(self, x):
        """Forward pass."""
        # Input shape: [batch_size, sequence_length, channels, height, width]
        batch_size = x.size(0)
        
        # Reshape for 3D CNN
        # Expected shape: [batch_size, channels, frames, height, width]
        x = x.permute(0, 2, 1, 3, 4)
        
        # Forward pass through backbone
        x = self.backbone(x)
        
        # Reshape output
        x = x.view(batch_size, -1)
        
        return x


class EmotionClassifier(nn.Module):
    """3D ResNet + LSTM model for emotion classification."""
    
    def __init__(self, num_classes=6, hidden_size=128, dropout=0.5):
        """Initialize the classifier."""
        super(EmotionClassifier, self).__init__()
        
        # Video embedder
        self.video_embedder = VideoEmbedder(pretrained=True, freeze_backbone=False)
        
        # Embedding size from 3D ResNet-18 (512)
        embedding_size = 512
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        """Forward pass."""
        # Input shape: [batch_size, sequence_length, channels, height, width]
        batch_size = x.size(0)
        sequence_length = x.size(1)
        
        # Process each frame in the sequence through the video embedder
        embeddings = []
        for t in range(sequence_length):
            frame = x[:, t]
            embedding = self.video_embedder(frame.unsqueeze(1))
            embeddings.append(embedding)
        
        # Stack embeddings to form a sequence
        embeddings = torch.stack(embeddings, dim=1)
        
        # Process sequence with LSTM
        lstm_out, _ = self.lstm(embeddings)
        
        # Take the output from the last time step
        lstm_final = lstm_out[:, -1]
        
        # Classify
        logits = self.classifier(lstm_final)
        
        return logits


def train(model, train_loader, optimizer, criterion, device, scaler=None, use_amp=False):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for frames, labels in pbar:
        frames = frames.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Use mixed precision if available
        if use_amp:
            with autocast():
                outputs = model(frames)
                loss = criterion(outputs, labels)
                
            # Scale gradients and optimize
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'acc': 100 * correct / total
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for frames, labels in tqdm(val_loader, desc="Validation"):
            frames = frames.to(device)
            labels = labels.to(device)
            
            outputs = model(frames)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    # Calculate metrics
    val_loss = running_loss / len(val_loader)
    val_acc = 100 * accuracy_score(all_labels, all_preds)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Classification report
    class_report = classification_report(
        all_labels, all_preds, 
        target_names=EMOTION_LABELS,
        output_dict=True
    )
    
    return val_loss, val_acc, cm, class_report


def plot_confusion_matrix(cm, output_dir, epoch):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTION_LABELS,
                yticklabels=EMOTION_LABELS)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    cm_file = os.path.join(output_dir, f'confusion_matrix_epoch_{epoch}.png')
    plt.tight_layout()
    plt.savefig(cm_file)
    plt.close()


def save_checkpoint(model, optimizer, scheduler, epoch, val_acc, output_dir, is_best=False):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_acc': val_acc
    }
    
    # Save regular checkpoint
    checkpoint_file = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_file)
    
    # Save best model if it's the best so far
    if is_best:
        best_model_file = os.path.join(output_dir, 'model_best.pt')
        torch.save(checkpoint, best_model_file)


def main(args):
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"video_full_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save args
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets and dataloaders
    train_dataset = VideoDataset(
        args.manifest_file,
        split='train',
        frames=args.frames,
        img_size=args.img_size,
        augment=True
    )
    
    val_dataset = VideoDataset(
        args.manifest_file,
        split='val',
        frames=args.frames,
        img_size=args.img_size,
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = EmotionClassifier(
        num_classes=len(EMOTION_LABELS),
        hidden_size=args.hidden_size,
        dropout=args.dropout
    )
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # LR scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    
    # Mixed precision
    use_amp = args.fp16 and torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    
    # Training loop
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, 
                                      device, scaler, use_amp)
        
        # Validate
        val_loss, val_acc, confusion_mat, class_report = validate(model, val_loader, 
                                                                 criterion, device)
        
        # Update LR
        scheduler.step()
        
        # Track metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Plot and save confusion matrix
        plot_confusion_matrix(confusion_mat, output_dir, epoch)
        
        # Save classification report
        with open(os.path.join(output_dir, f'classification_report_epoch_{epoch}.json'), 'w') as f:
            json.dump(class_report, f, indent=2)
        
        # Check if this is the best model
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            print(f"New best validation accuracy: {best_val_acc:.2f}%")
        
        # Save checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, val_acc, 
                       output_dir, is_best)
        
        # Save metrics
        metrics = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'best_val_acc': best_val_acc
        }
        
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Plot learning curves
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curves')
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Acc')
        plt.plot(val_accs, label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Accuracy Curves')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'learning_curves.png'))
        plt.close()
    
    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model and results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train 3D-ResNet-LSTM emotion classifier')
    
    # Data parameters
    parser.add_argument('--manifest_file', type=str, required=True,
                        help='Path to the manifest CSV file')
    parser.add_argument('--frames', type=int, default=48,
                        help='Number of frames to use from each video')
    parser.add_argument('--img_size', type=int, default=112,
                        help='Size to resize video frames to (square)')
    
    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='Hidden size for LSTM')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--fp16', action='store_true',
                        help='Use mixed precision training')
    
    # Other parameters
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--output_dir', type=str, default='/home/ubuntu/emotion_full_video',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    main(args)
