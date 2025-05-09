#!/usr/bin/env python3
"""
Examine a single NPZ file from the RAVDESS features directory
to understand its detailed structure and content.
"""

import os
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt

def examine_npz_file(file_path, visualize=False, output_dir=None):
    """
    Examine the contents of a single NPZ file in detail
    and print a comprehensive summary of its structure.
    
    Args:
        file_path: Path to the NPZ file
        visualize: Whether to generate visualizations
        output_dir: Directory to save visualizations
    """
    try:
        print(f"\n{'='*80}\nExamining: {os.path.basename(file_path)}\n{'='*80}")
        
        # Load the NPZ file
        data = np.load(file_path, allow_pickle=True)
        
        # List all keys in the file
        print(f"Keys in the file: {list(data.keys())}")
        
        # Process each key
        for key in data.keys():
            print(f"\n--- Key: {key} ---")
            
            # Get the array for this key
            array = data[key]
            
            # Special handling for params (typically stored as an object array)
            if key == 'params' and array.dtype == np.dtype('O'):
                try:
                    params = array.item()
                    if isinstance(params, dict):
                        print("Parameters:")
                        for param_key, param_value in params.items():
                            print(f"  {param_key}: {param_value}")
                    else:
                        print(f"Params (type {type(params)}): {params}")
                except Exception as e:
                    print(f"Error unpacking params: {e}")
                continue
            
            # Special handling for scalar values stored as object arrays
            if array.dtype == np.dtype('O') and array.size == 1:
                try:
                    value = array.item()
                    print(f"Value (type {type(value)}): {value}")
                except Exception as e:
                    print(f"Error unpacking object: {e}")
                continue
            
            # Print array shape and data type
            print(f"Shape: {array.shape}")
            print(f"Data type: {array.dtype}")
            
            # Skip further analysis for non-numeric data
            if not np.issubdtype(array.dtype, np.number):
                print("Non-numeric data, skipping statistical analysis")
                continue
            
            # Basic statistics for numeric arrays
            if array.size > 0:
                print(f"Min: {np.min(array):.4f}")
                print(f"Max: {np.max(array):.4f}")
                print(f"Mean: {np.mean(array):.4f}")
                print(f"Std: {np.std(array):.4f}")
                print(f"Non-zero elements: {np.mean(array != 0) * 100:.1f}%")
                
                # Additional statistics for large arrays
                if array.size > 1000:
                    # Percentiles
                    percentiles = [0, 5, 25, 50, 75, 95, 100]
                    percentile_values = np.percentile(array.flatten(), percentiles)
                    print("Percentiles:")
                    for p, v in zip(percentiles, percentile_values):
                        print(f"  {p}%: {v:.4f}")
                    
                    # Check for NaN and Inf values
                    nan_count = np.isnan(array).sum()
                    inf_count = np.isinf(array).sum()
                    print(f"NaN values: {nan_count}")
                    print(f"Inf values: {inf_count}")
            
            # Generate visualizations for key arrays if requested
            if visualize and output_dir and array.size > 0:
                if key == 'video_features':
                    visualize_features(array, key, output_dir)
                elif key == 'audio_features':
                    visualize_features(array, key, output_dir)
    
    except Exception as e:
        print(f"Error examining {file_path}: {e}")
        import traceback
        traceback.print_exc()

def visualize_features(features, feature_type, output_dir):
    """
    Create visualizations for feature arrays.
    
    Args:
        features: Feature array to visualize
        feature_type: Type of features ('video_features' or 'audio_features')
        output_dir: Directory to save visualizations
    """
    base_filename = os.path.join(output_dir, feature_type)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Histogram of feature values
    plt.figure(figsize=(10, 6))
    plt.hist(features.flatten(), bins=100, alpha=0.7)
    plt.title(f'Distribution of {feature_type.replace("_", " ").title()} Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{base_filename}_histogram.png")
    plt.close()
    
    # 2. Feature means per frame
    feature_means = np.mean(features, axis=1)
    plt.figure(figsize=(12, 6))
    plt.plot(feature_means)
    plt.title(f'Mean Feature Value per Frame ({feature_type.replace("_", " ").title()})')
    plt.xlabel('Frame Number')
    plt.ylabel('Mean Feature Value')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{base_filename}_means.png")
    plt.close()
    
    # 3. Feature standard deviations per frame
    feature_stds = np.std(features, axis=1)
    plt.figure(figsize=(12, 6))
    plt.plot(feature_stds)
    plt.title(f'Feature Standard Deviation per Frame ({feature_type.replace("_", " ").title()})')
    plt.xlabel('Frame Number')
    plt.ylabel('Feature Std Dev')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{base_filename}_stds.png")
    plt.close()
    
    # 4. Heatmap of features (limit to first 10 frames for clarity)
    num_frames = min(10, features.shape[0])
    num_features = min(50, features.shape[1])  # Show at most 50 features
    
    plt.figure(figsize=(12, 8))
    plt.imshow(features[:num_frames, :num_features], aspect='auto', cmap='viridis')
    plt.colorbar(label='Feature Value')
    plt.title(f'{feature_type.replace("_", " ").title()} Heatmap (First {num_frames} Frames, First {num_features} Features)')
    plt.xlabel('Feature Index')
    plt.ylabel('Frame Number')
    plt.savefig(f"{base_filename}_heatmap.png")
    plt.close()
    
    # 5. Visualize temporal patterns with a line plot of select features
    selected_features = [0, 1, 2, 3, 4]  # First 5 features
    plt.figure(figsize=(12, 8))
    
    for i in selected_features:
        if i < features.shape[1]:
            plt.plot(features[:, i], label=f'Feature {i}')
    
    plt.title(f'Temporal Patterns of Select {feature_type.replace("_", " ").title()}')
    plt.xlabel('Frame Number')
    plt.ylabel('Feature Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{base_filename}_temporal.png")
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Examine a single NPZ file in detail")
    parser.add_argument("file_path", help="Path to the NPZ file to examine")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--output-dir", default="npz_visualizations", help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.file_path):
        print(f"Error: File '{args.file_path}' does not exist.")
        return
    
    examine_npz_file(args.file_path, args.visualize, args.output_dir)

if __name__ == "__main__":
    main()
