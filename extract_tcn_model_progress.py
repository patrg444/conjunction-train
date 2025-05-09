#!/usr/bin/env python3
"""
Script to extract validation accuracy from branched_regularization_sync_aug_tcn logs
and plot/report progress by epoch.
"""

import re
import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def extract_metrics(log_file):
    """Extract epoch, loss, accuracy, val_loss, val_accuracy from training logs."""
    metrics = defaultdict(list)
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Pattern to match epoch results
    pattern = r'Epoch (\d+)/\d+.*?loss: ([\d\.]+).*?accuracy: ([\d\.]+).*?val_loss: ([\d\.]+).*?val_accuracy: ([\d\.]+)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for match in matches:
        epoch = int(match[0])
        metrics['epoch'].append(epoch)
        metrics['loss'].append(float(match[1]))
        metrics['accuracy'].append(float(match[2]))
        metrics['val_loss'].append(float(match[3]))
        metrics['val_accuracy'].append(float(match[4]))
    
    return metrics

def plot_metrics(metrics, output_file='tcn_model_progress.png'):
    """Plot training and validation metrics."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot accuracy
    ax1.plot(metrics['epoch'], metrics['accuracy'], 'b-', label='Training Accuracy')
    ax1.plot(metrics['epoch'], metrics['val_accuracy'], 'r-', label='Validation Accuracy')
    ax1.set_title('Model Accuracy by Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(metrics['epoch'], metrics['loss'], 'b-', label='Training Loss')
    ax2.plot(metrics['epoch'], metrics['val_loss'], 'r-', label='Validation Loss')
    ax2.set_title('Model Loss by Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Progress plot saved to {output_file}")

def print_progress_table(metrics):
    """Print a formatted table of training progress."""
    print("\nTraining Progress for Branched Regularization Sync Aug TCN Model:")
    print("-" * 80)
    print(f"{'Epoch':^10}{'Train Loss':^15}{'Train Acc':^15}{'Val Loss':^15}{'Val Acc':^15}")
    print("-" * 80)
    
    for i in range(len(metrics['epoch'])):
        print(f"{metrics['epoch'][i]:^10}{metrics['loss'][i]:^15.4f}{metrics['accuracy'][i]:^15.4f}{metrics['val_loss'][i]:^15.4f}{metrics['val_accuracy'][i]:^15.4f}")
    
    if metrics['epoch']:
        latest_epoch = max(metrics['epoch'])
        latest_idx = metrics['epoch'].index(latest_epoch)
        best_val_acc = max(metrics['val_accuracy'])
        best_idx = metrics['val_accuracy'].index(best_val_acc)
        
        print("-" * 80)
        print(f"Latest validation accuracy (Epoch {latest_epoch}): {metrics['val_accuracy'][latest_idx]:.4f} ({metrics['val_accuracy'][latest_idx]*100:.2f}%)")
        print(f"Best validation accuracy so far (Epoch {metrics['epoch'][best_idx]}): {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
        
        # If we have enough data, calculate trend
        if len(metrics['val_accuracy']) >= 3:
            recent_trend = np.polyfit(metrics['epoch'][-3:], metrics['val_accuracy'][-3:], 1)[0]
            print(f"Recent trend (slope over last 3 epochs): {recent_trend:.6f}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_tcn_model_progress.py <log_file>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    metrics = extract_metrics(log_file)
    
    if not metrics['epoch']:
        print("No training data found in log file")
        sys.exit(1)
    
    print_progress_table(metrics)
    plot_metrics(metrics)

if __name__ == "__main__":
    main()
