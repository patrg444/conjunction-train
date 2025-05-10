#!/usr/bin/env python3
"""
Extract model weights from the full checkpoint and save to a smaller file.
This removes optimizer state, scheduler state, and other training metadata,
reducing file size significantly for deployment.
"""

import os
import torch
import argparse

def extract_model_weights(input_path, output_path):
    print(f"Loading checkpoint from: {input_path}")
    checkpoint = torch.load(input_path, map_location='cpu')
    
    if 'model_state_dict' not in checkpoint:
        print("Error: Could not find model_state_dict in checkpoint")
        return False
    
    model_state_dict = checkpoint['model_state_dict']
    val_acc = checkpoint.get('val_acc', 'unknown')
    
    print(f"Validation accuracy from checkpoint: {val_acc}")
    print(f"Model has {len(model_state_dict)} layers")
    
    # Save just the model weights
    print(f"Saving model weights to: {output_path}")
    torch.save(model_state_dict, output_path)
    
    # Check file sizes
    input_size = os.path.getsize(input_path) / (1024 * 1024)
    output_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"Original checkpoint size: {input_size:.2f} MB")
    print(f"Extracted model size: {output_size:.2f} MB")
    print(f"Size reduction: {(1 - output_size/input_size) * 100:.2f}%")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract model weights from checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Path to full checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Path to save model weights")
    args = parser.parse_args()
    
    extract_model_weights(args.input, args.output)
