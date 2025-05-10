#!/usr/bin/env python3
"""
Script to extract validation accuracy from all model training logs on AWS.
This script connects to the AWS instance, finds all training logs, and extracts
the validation accuracy metrics for each model.

Usage:
    python extract_all_models_val_accuracy.py [--plot]

Options:
    --plot       Generate accuracy plots for each model
"""

import re
import os
import sys
import subprocess
import json
import csv
import argparse
import glob # Added glob
import matplotlib.pyplot as plt # type: ignore
import numpy as np
from collections import defaultdict
from datetime import datetime

# --- Removed AWS connection details and SSH/SCP functions ---

# Define the local directory where logs are expected on the EC2 instance
LOG_DIRECTORY_ON_EC2 = "/home/ec2-user/emotion_training/logs"


def extract_model_name(log_path):
    """Extract a user-friendly model name from the log file path."""
    basename = os.path.basename(log_path)
    # Remove 'training_' prefix and '.log' suffix if present
    model_name = basename.replace("training_", "").replace(".log", "")
    return model_name

def extract_metrics(log_content):
    """Extract epoch, loss, accuracy, val_loss, val_accuracy from log content."""
    metrics = defaultdict(list)
    # New pattern to match lines like:
    # 579/579 [...] - loss: 1.3666 - accuracy: 0.6677 - val_loss: 1.3361 - val_accuracy: 0.6759
    pattern = r'loss: ([\d\.]+) - accuracy: ([\d\.]+) - val_loss: ([\d\.]+) - val_accuracy: ([\d\.]+)'
    # Also look for preceding "Epoch N" or "Epoch N/M" lines to get epoch number
    epoch_pattern = r'Epoch\s+(\d+)(?:/\d+)?'
    lines = log_content.splitlines()
    current_epoch = None
    for i, line in enumerate(lines):
        epoch_match = re.match(epoch_pattern, line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
        result = re.search(pattern, line)
        if result:
            # Use the most recent epoch number seen above this line
            epoch = current_epoch if current_epoch is not None else len(metrics['epoch']) + 1
            metrics['epoch'].append(epoch)
            metrics['loss'].append(float(result.group(1)))
            metrics['accuracy'].append(float(result.group(2)))
            metrics['val_loss'].append(float(result.group(3)))
            metrics['val_accuracy'].append(float(result.group(4)))
    return metrics

def extract_metrics_from_file(local_log_file, tail_lines=400):
    """Read only the last tail_lines of a log file and extract metrics."""
    try:
        # Efficiently read the last N lines of the file
        with open(local_log_file, 'rb') as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            block_size = 4096
            data = b''
            lines_found = 0
            while file_size > 0 and lines_found < tail_lines:
                read_size = min(block_size, file_size)
                f.seek(file_size - read_size, os.SEEK_SET)
                data = f.read(read_size) + data
                file_size -= read_size
                lines_found = data.count(b'\n')
            # Decode and split into lines, take the last tail_lines
            lines = data.decode(errors='replace').splitlines()[-tail_lines:]
            content = '\n'.join(lines)
        return extract_metrics(content)
    except Exception as e:
        print(f"Error processing {local_log_file}: {e}")
        return defaultdict(list)

def get_model_summary(metrics):
    """Generate a summary of model training metrics."""
    if not metrics['epoch']:
        return {
            'epochs_completed': 0,
            'latest_val_accuracy': None,
            'best_val_accuracy': None,
            'best_epoch': None
        }
    
    best_val_acc = max(metrics['val_accuracy'])
    best_idx = metrics['val_accuracy'].index(best_val_acc)
    best_epoch = metrics['epoch'][best_idx]
    
    # Get the latest epoch metrics
    latest_epoch = max(metrics['epoch'])
    latest_idx = metrics['epoch'].index(latest_epoch)
    latest_val_acc = metrics['val_accuracy'][latest_idx]
    
    # Calculate recent trend if possible
    if len(metrics['val_accuracy']) >= 3:
        recent_epochs = metrics['epoch'][-3:]
        recent_accuracies = metrics['val_accuracy'][-3:]
        trend = np.polyfit(recent_epochs, recent_accuracies, 1)[0]
    else:
        trend = None
    
    return {
        'epochs_completed': latest_epoch,
        'latest_val_accuracy': latest_val_acc,
        'best_val_accuracy': best_val_acc,
        'best_epoch': best_epoch,
        'recent_trend': trend
    }

def plot_model_accuracy(model_name, metrics, output_dir="model_accuracy_plots"):
    """Generate a plot of training and validation accuracy for a model."""
    if not metrics['epoch']:
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Plot accuracy
    plt.plot(metrics['epoch'], metrics['accuracy'], 'b-', label='Training Accuracy')
    plt.plot(metrics['epoch'], metrics['val_accuracy'], 'r-', label='Validation Accuracy')
    
    # Add gridlines for readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Training and Validation Accuracy: {model_name}', fontsize=14)
    
    # Add legend
    plt.legend(fontsize=10)
    
    # Add best validation accuracy marker
    best_val_acc = max(metrics['val_accuracy'])
    best_idx = metrics['val_accuracy'].index(best_val_acc)
    best_epoch = metrics['epoch'][best_idx]
    
    plt.scatter(best_epoch, best_val_acc, color='green', s=100, zorder=5)
    plt.annotate(f'Best: {best_val_acc:.4f} (Epoch {best_epoch})', 
                 (best_epoch, best_val_acc),
                 xytext=(10, -20),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', color='green'))
    
    # Set y-axis to start at 0 and extend slightly above the max accuracy
    plt.ylim([0, max(max(metrics['accuracy']), max(metrics['val_accuracy'])) * 1.05])
    
    # Save the figure
    output_file = os.path.join(output_dir, f"{model_name}_accuracy.png")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"Generated accuracy plot for {model_name}: {output_file}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract validation accuracy from training logs')
    parser.add_argument('log_dir', nargs='?', default=LOG_DIRECTORY_ON_EC2, 
                        help=f'Optional path to the local log directory to scan (defaults to {LOG_DIRECTORY_ON_EC2})')
    parser.add_argument('--plot', action='store_true', help='Generate accuracy plots for each model')
    args = parser.parse_args()

    print("=" * 80)
    print(f"EXTRACTING VALIDATION ACCURACY FROM LOCAL LOGS in: {args.log_dir}")
    print("=" * 80)

    all_models = {}
    log_directory = args.log_dir

    if not os.path.isdir(log_directory):
        print(f"Error: Log directory not found: {log_directory}")
        sys.exit(1)

    # Find log files locally using glob
    log_pattern = os.path.join(log_directory, '*.log')
    local_log_files = glob.glob(log_pattern)
    
    if not local_log_files:
        print(f"No .log files found in {log_directory}.")
        return

    print(f"Found {len(local_log_files)} local log files.")

    # Process each local log file
    for local_file_path in local_log_files:
        # Skip temporary files if any exist with that pattern
        if '_tmp' in os.path.basename(local_file_path):
            continue
        # Skip known problematic file(s)
        if os.path.basename(local_file_path) == "spectrogram_cnn_lstm_20250331_014144.log":
            print(f"Skipping known problematic file: {local_file_path}")
            continue

        model_name = extract_model_name(local_file_path)
        print(f"Processing {os.path.basename(local_file_path)} as model {model_name}...")

        try:
            metrics = extract_metrics_from_file(local_file_path)
            if metrics['epoch']:
                summary = get_model_summary(metrics)
                all_models[model_name] = {
                    'log_file': local_file_path, # Store local path
                    'summary': summary,
                    'metrics': {
                        'epoch': metrics['epoch'],
                        'val_accuracy': metrics['val_accuracy']
                    }
                }
                if args.plot:
                    # Ensure plots are saved locally relative to script execution
                    plot_model_accuracy(model_name, metrics, output_dir="model_accuracy_plots_local") 
                print(f"Finished analysis for {model_name}.")
            else:
                print(f"Warning: No epoch data found in log file for {model_name}. Skipping.")
                print(f"Finished analysis for {model_name}.")
        except Exception as e:
            print(f"Error processing log file {local_file_path} for {model_name}: {e}")
            print(f"Finished analysis for {model_name} (with error).")

    # Define output file names (relative to script execution dir)
    output_json = "model_validation_accuracy_local.json"
    output_csv = "model_validation_summary_local.csv"

    # --- Common processing ---
    if not all_models:
        print("No models processed or no data found.")
        return

    # Save results to JSON
    try:
        with open(output_json, 'w') as f:
            json.dump(all_models, f, indent=2)
        print(f"Saved detailed results to {output_json}")
    except IOError as e:
        print(f"Error saving JSON results to {output_json}: {e}")

    # Generate CSV summary
    try:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow([
                'Model', 'Epochs Completed', 'Latest Val Accuracy', 
                'Best Val Accuracy', 'Best Epoch', 'Trend'
            ])
            
            # Sort models by best validation accuracy
            sorted_models = sorted(
                all_models.items(), 
                key=lambda item: item[1]['summary']['best_val_accuracy'] if item[1]['summary'] and item[1]['summary']['best_val_accuracy'] is not None else 0,
                reverse=True
            )
            
            # Write data
            for model_name, data in sorted_models:
                summary = data.get('summary', {})
                writer.writerow([
                    model_name,
                    summary.get('epochs_completed', 'N/A'),
                    f"{summary.get('latest_val_accuracy', 'N/A'):.4f}" if isinstance(summary.get('latest_val_accuracy'), float) else "N/A",
                    f"{summary.get('best_val_accuracy', 'N/A'):.4f}" if isinstance(summary.get('best_val_accuracy'), float) else "N/A",
                    summary.get('best_epoch', 'N/A'),
                    f"{summary.get('recent_trend', 'N/A'):.6f}" if isinstance(summary.get('recent_trend'), float) else "N/A"
                ])
        
        print(f"Saved summary to {output_csv}")
    except IOError as e:
        print(f"Error saving CSV summary to {output_csv}: {e}")

    # Print summary table
    print("\nSUMMARY OF MODEL VALIDATION ACCURACY")
    print("=" * 80)
    # Use the same sorted_models list from CSV generation
    if 'sorted_models' in locals():
        print(f"{'Model':<50} {'Best Val Acc':<20} {'Epoch':<10}")
        print("-" * 80)
        
        for model_name, data in sorted_models:
            summary = data.get('summary', {})
            best_acc = summary.get('best_val_accuracy')
            best_epoch = summary.get('best_epoch', 'N/A')
            if isinstance(best_acc, float):
                print(f"{model_name:<50} {best_acc:.4f} ({best_acc*100:.2f}%)     {best_epoch:<10}")
            else:
                 print(f"{model_name:<50} {'N/A':<20} {best_epoch:<10}")

        print("=" * 80)
        print(f"Detailed history available in {output_json}")
        print(f"Summary table available in {output_csv}")
    else:
        print("No models found or processed to display summary.")
        print("=" * 80)


if __name__ == "__main__":
    main()
