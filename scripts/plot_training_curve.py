#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility script to plot training curves from saved model history.
Visualizes accuracy and loss over epochs, useful for analyzing model performance.
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import glob
from datetime import datetime

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Plot training curves from model history')
    
    parser.add_argument('--history_file', type=str, default=None,
                        help='Path to saved history JSON file')
    parser.add_argument('--dir', type=str, default=None,
                        help='Directory containing history JSON files (will plot most recent)')
    parser.add_argument('--metric', type=str, default='both',
                        choices=['accuracy', 'loss', 'both', 'all'],
                        help='Which metric to plot (accuracy, loss, both, or all including custom metrics)')
    parser.add_argument('--save', action='store_true',
                        help='Save plots to PNG files instead of displaying')
    parser.add_argument('--title', type=str, default=None,
                        help='Custom title for the plot')
    parser.add_argument('--smooth', type=int, default=0,
                        help='Apply moving average smoothing with specified window size')
    
    return parser.parse_args()

def moving_average(data, window_size):
    """Apply moving average smoothing to a data series."""
    if window_size <= 1:
        return data
    
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')

def plot_history(history_dict, metric='both', title=None, save=False, smooth=0):
    """
    Plot training curves from history dictionary.
    
    Args:
        history_dict: Dictionary containing training history
        metric: Which metric to plot ('accuracy', 'loss', 'both', or 'all')
        title: Custom title for the plot
        save: Whether to save the plot as a PNG file
        smooth: Window size for moving average smoothing
    """
    # Ensure we have epochs data
    epochs = range(1, len(history_dict.get('loss', [])) + 1)
    if not epochs:
        print("Error: No training data found in history")
        return
    
    # Create output directory for saving plots
    if save:
        os.makedirs('training_plots', exist_ok=True)
    
    # Current timestamp for saving files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Function to apply smoothing if needed
    def smooth_data(data):
        if smooth > 1 and len(data) > smooth:
            return moving_average(data, smooth)
        return data
    
    # Function to adjust epochs for smoothed data
    def adjust_epochs(epochs, data):
        if smooth > 1 and len(data) > smooth:
            return epochs[(smooth-1)//2:-(smooth-1)//2] if smooth % 2 == 1 else epochs[smooth//2-1:-smooth//2]
        return epochs
    
    # Plot accuracy
    if metric in ['accuracy', 'both', 'all']:
        if 'accuracy' in history_dict and 'val_accuracy' in history_dict:
            plt.figure(figsize=(10, 6))
            
            # Get data
            train_acc = history_dict['accuracy']
            val_acc = history_dict['val_accuracy']
            
            # Apply smoothing if needed
            train_acc_smooth = smooth_data(train_acc)
            val_acc_smooth = smooth_data(val_acc)
            
            # Adjust epochs for smoothed data
            epochs_smooth_train = adjust_epochs(epochs, train_acc)
            epochs_smooth_val = adjust_epochs(epochs, val_acc)
            
            # Plot
            plt.plot(epochs_smooth_train, train_acc_smooth, 'b', label='Training accuracy')
            plt.plot(epochs_smooth_val, val_acc_smooth, 'r', label='Validation accuracy')
            
            # Add plot details
            plot_title = title if title else 'Training and Validation Accuracy'
            plt.title(plot_title)
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            
            # Add final values annotation
            final_train_acc = train_acc[-1]
            final_val_acc = val_acc[-1]
            max_val_acc = max(val_acc)
            max_val_epoch = val_acc.index(max_val_acc) + 1
            
            plt.annotate(f'Final train: {final_train_acc:.4f}',
                         xy=(len(epochs), final_train_acc),
                         xytext=(0.7*len(epochs), 0.2),
                         arrowprops=dict(facecolor='blue', shrink=0.05))
            
            plt.annotate(f'Final val: {final_val_acc:.4f}',
                         xy=(len(epochs), final_val_acc),
                         xytext=(0.7*len(epochs), 0.1),
                         arrowprops=dict(facecolor='red', shrink=0.05))
            
            plt.annotate(f'Best val: {max_val_acc:.4f} (epoch {max_val_epoch})',
                         xy=(max_val_epoch, max_val_acc),
                         xytext=(0.3*len(epochs), 0.9),
                         arrowprops=dict(facecolor='green', shrink=0.05))
            
            # Save or show plot
            if save:
                filename = f'training_plots/accuracy_{timestamp}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Saved accuracy plot to {filename}")
            else:
                plt.show()
    
    # Plot loss
    if metric in ['loss', 'both', 'all']:
        if 'loss' in history_dict and 'val_loss' in history_dict:
            plt.figure(figsize=(10, 6))
            
            # Get data
            train_loss = history_dict['loss']
            val_loss = history_dict['val_loss']
            
            # Apply smoothing if needed
            train_loss_smooth = smooth_data(train_loss)
            val_loss_smooth = smooth_data(val_loss)
            
            # Adjust epochs for smoothed data
            epochs_smooth_train = adjust_epochs(epochs, train_loss)
            epochs_smooth_val = adjust_epochs(epochs, val_loss)
            
            # Plot
            plt.plot(epochs_smooth_train, train_loss_smooth, 'b', label='Training loss')
            plt.plot(epochs_smooth_val, val_loss_smooth, 'r', label='Validation loss')
            
            # Add plot details
            plot_title = title if title else 'Training and Validation Loss'
            plt.title(plot_title)
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            
            # Add final values annotation
            final_train_loss = train_loss[-1]
            final_val_loss = val_loss[-1]
            min_val_loss = min(val_loss)
            min_val_epoch = val_loss.index(min_val_loss) + 1
            
            plt.annotate(f'Final train: {final_train_loss:.4f}',
                         xy=(len(epochs), final_train_loss),
                         xytext=(0.7*len(epochs), 0.8),
                         arrowprops=dict(facecolor='blue', shrink=0.05))
            
            plt.annotate(f'Final val: {final_val_loss:.4f}',
                         xy=(len(epochs), final_val_loss),
                         xytext=(0.7*len(epochs), 0.9),
                         arrowprops=dict(facecolor='red', shrink=0.05))
            
            plt.annotate(f'Best val: {min_val_loss:.4f} (epoch {min_val_epoch})',
                         xy=(min_val_epoch, min_val_loss),
                         xytext=(0.3*len(epochs), 0.2),
                         arrowprops=dict(facecolor='green', shrink=0.05))
            
            # Save or show plot
            if save:
                filename = f'training_plots/loss_{timestamp}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Saved loss plot to {filename}")
            else:
                plt.show()
    
    # Plot other metrics if requested
    if metric == 'all':
        for key in history_dict.keys():
            # Skip already plotted metrics and ones without val_ counterpart
            if key in ['accuracy', 'loss'] or key.startswith('val_') or f'val_{key}' not in history_dict:
                continue
                
            plt.figure(figsize=(10, 6))
            
            # Get data
            train_metric = history_dict[key]
            val_metric = history_dict[f'val_{key}']
            
            # Apply smoothing if needed
            train_metric_smooth = smooth_data(train_metric)
            val_metric_smooth = smooth_data(val_metric)
            
            # Adjust epochs for smoothed data
            epochs_smooth_train = adjust_epochs(epochs, train_metric)
            epochs_smooth_val = adjust_epochs(epochs, val_metric)
            
            # Plot
            plt.plot(epochs_smooth_train, train_metric_smooth, 'b', label=f'Training {key}')
            plt.plot(epochs_smooth_val, val_metric_smooth, 'r', label=f'Validation {key}')
            
            # Add plot details
            plot_title = title if title else f'Training and Validation {key.capitalize()}'
            plt.title(plot_title)
            plt.xlabel('Epochs')
            plt.ylabel(key.capitalize())
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            
            # Save or show plot
            if save:
                filename = f'training_plots/{key}_{timestamp}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Saved {key} plot to {filename}")
            else:
                plt.show()

def find_most_recent_history_file(directory):
    """Find the most recent history file in a directory."""
    pattern = os.path.join(directory, '*history*.json')
    history_files = glob.glob(pattern)
    
    if not history_files:
        return None
    
    # Sort by modification time, newest first
    history_files.sort(key=os.path.getmtime, reverse=True)
    return history_files[0]

def main():
    args = parse_arguments()
    
    # Determine history file to use
    history_file = None
    if args.history_file:
        history_file = args.history_file
    elif args.dir:
        history_file = find_most_recent_history_file(args.dir)
        if history_file:
            print(f"Using most recent history file: {history_file}")
    
    if not history_file:
        print("Error: No history file specified or found.")
        print("Please provide either --history_file or --dir arguments.")
        print("Example usage:")
        print("  python plot_training_curve.py --history_file checkpoints/model_history.json")
        print("  python plot_training_curve.py --dir checkpoints --metric both --save")
        return
    
    # Load history file
    try:
        with open(history_file, 'r') as f:
            history_dict = json.load(f)
        
        print(f"Loaded history from {history_file}")
        print(f"Available metrics: {', '.join(key for key in history_dict.keys() if not key.startswith('val_'))}")
        print(f"Training epochs: {len(history_dict.get('loss', []))}")
        
        # Plot training history
        plot_history(
            history_dict=history_dict,
            metric=args.metric,
            title=args.title,
            save=args.save,
            smooth=args.smooth
        )
        
    except Exception as e:
        print(f"Error processing history file: {e}")

if __name__ == "__main__":
    main()
