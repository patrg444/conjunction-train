#!/usr/bin/env python3
"""
This script merges multiple humor dataset manifests into a single consolidated manifest.
It combines datasets like UR-FUNNY, Short-Humor, and any other humor datasets
while ensuring proper balance between humorous and non-humorous examples.
"""

import os
import pandas as pd
import argparse
from tqdm import tqdm
import numpy as np

def setup_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs("datasets/manifests/humor", exist_ok=True)

def merge_humor_manifests(manifests, output_prefix="combined", balance=True):
    """
    Merge multiple humor dataset manifests into consolidated train/val manifests
    
    Args:
        manifests: List of dictionaries with 'train' and 'val' paths for each dataset
        output_prefix: Prefix for output files (default: "combined")
        balance: Whether to balance positive and negative examples (default: True)
    
    Returns:
        Dictionary with paths to the merged manifests
    """
    # Initialize empty lists for train and val data
    train_data = []
    val_data = []
    
    print("Merging humor manifests:")
    
    # Track statistics for reporting
    dataset_stats = {}
    
    # Process each dataset
    for dataset_info in manifests:
        dataset_name = dataset_info.get('name', 'Unknown')
        dataset_stats[dataset_name] = {'train': {}, 'val': {}}
        
        # Process train manifest
        train_path = dataset_info.get('train')
        if train_path and os.path.exists(train_path):
            print(f"  Processing train manifest: {train_path}")
            train_df = pd.read_csv(train_path)
            
            # Add source column if not present
            if 'source' not in train_df.columns:
                train_df['source'] = dataset_name
            
            # Track statistics
            dataset_stats[dataset_name]['train']['total'] = len(train_df)
            dataset_stats[dataset_name]['train']['humor'] = sum(train_df['label'] == 1)
            dataset_stats[dataset_name]['train']['non_humor'] = sum(train_df['label'] == 0)
            
            train_data.append(train_df)
        
        # Process validation manifest
        val_path = dataset_info.get('val')
        if val_path and os.path.exists(val_path):
            print(f"  Processing validation manifest: {val_path}")
            val_df = pd.read_csv(val_path)
            
            # Add source column if not present
            if 'source' not in val_df.columns:
                val_df['source'] = dataset_name
            
            # Track statistics
            dataset_stats[dataset_name]['val']['total'] = len(val_df)
            dataset_stats[dataset_name]['val']['humor'] = sum(val_df['label'] == 1)
            dataset_stats[dataset_name]['val']['non_humor'] = sum(val_df['label'] == 0)
            
            val_data.append(val_df)
    
    # Combine all datasets
    if train_data:
        train_combined = pd.concat(train_data, ignore_index=True)
    else:
        train_combined = pd.DataFrame()
    
    if val_data:
        val_combined = pd.concat(val_data, ignore_index=True)
    else:
        val_combined = pd.DataFrame()
    
    # Balance the datasets if requested
    if balance and not train_combined.empty:
        print("Balancing train dataset...")
        
        # Group by source and label
        grouped = train_combined.groupby(['source', 'label'])
        
        # Find the minimum count of each class across all sources
        humor_counts = [len(group) for (source, label), group in grouped if label == 1]
        non_humor_counts = [len(group) for (source, label), group in grouped if label == 0]
        
        min_humor = min(humor_counts) if humor_counts else 0
        min_non_humor = min(non_humor_counts) if non_humor_counts else 0
        
        # Determine target counts for balancing
        target_per_class = max(min_humor, min_non_humor)
        target_per_class = max(target_per_class, 1000)  # Ensure at least 1000 samples per class
        
        # Sample from each group
        balanced_dfs = []
        
        for (source, label), group in grouped:
            if len(group) > target_per_class:
                # Downsample
                sampled = group.sample(n=target_per_class, random_state=42)
            else:
                # Upsample if significantly smaller
                if len(group) < target_per_class * 0.5:
                    sampled = group.sample(n=target_per_class, replace=True, random_state=42)
                else:
                    sampled = group
            
            balanced_dfs.append(sampled)
        
        # Combine balanced groups
        train_combined = pd.concat(balanced_dfs, ignore_index=True)
    
    # Same balancing for validation if needed
    if balance and not val_combined.empty:
        print("Balancing validation dataset...")
        
        # Group by source and label
        grouped = val_combined.groupby(['source', 'label'])
        
        # Find the minimum count of each class across all sources
        humor_counts = [len(group) for (source, label), group in grouped if label == 1]
        non_humor_counts = [len(group) for (source, label), group in grouped if label == 0]
        
        min_humor = min(humor_counts) if humor_counts else 0
        min_non_humor = min(non_humor_counts) if non_humor_counts else 0
        
        # Determine target counts for balancing
        target_per_class = max(min_humor, min_non_humor)
        target_per_class = max(target_per_class, 200)  # Ensure at least 200 samples per class for validation
        
        # Sample from each group
        balanced_dfs = []
        
        for (source, label), group in grouped:
            if len(group) > target_per_class:
                # Downsample
                sampled = group.sample(n=target_per_class, random_state=42)
            else:
                # Upsample if significantly smaller
                if len(group) < target_per_class * 0.5:
                    sampled = group.sample(n=target_per_class, replace=True, random_state=42)
                else:
                    sampled = group
            
            balanced_dfs.append(sampled)
        
        # Combine balanced groups
        val_combined = pd.concat(balanced_dfs, ignore_index=True)
    
    # Shuffle the data
    if not train_combined.empty:
        train_combined = train_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    if not val_combined.empty:
        val_combined = val_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save the merged manifests
    output_files = {}
    
    if not train_combined.empty:
        train_output = f"datasets/manifests/humor/{output_prefix}_train_humor.csv"
        train_combined.to_csv(train_output, index=False)
        output_files["train"] = train_output
        
        print(f"\nCreated merged train manifest: {train_output}")
        print(f"  Total samples: {len(train_combined)}")
        print(f"  Label 0 (non-humorous): {sum(train_combined['label'] == 0)}")
        print(f"  Label 1 (humorous): {sum(train_combined['label'] == 1)}")
    
    if not val_combined.empty:
        val_output = f"datasets/manifests/humor/{output_prefix}_val_humor.csv"
        val_combined.to_csv(val_output, index=False)
        output_files["val"] = val_output
        
        print(f"\nCreated merged validation manifest: {val_output}")
        print(f"  Total samples: {len(val_combined)}")
        print(f"  Label 0 (non-humorous): {sum(val_combined['label'] == 0)}")
        print(f"  Label 1 (humorous): {sum(val_combined['label'] == 1)}")
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    for dataset_name, stats in dataset_stats.items():
        print(f"\n{dataset_name}:")
        
        if 'train' in stats and stats['train']:
            print(f"  Training:")
            print(f"    Total: {stats['train'].get('total', 0)}")
            print(f"    Humor: {stats['train'].get('humor', 0)}")
            print(f"    Non-humor: {stats['train'].get('non_humor', 0)}")
        
        if 'val' in stats and stats['val']:
            print(f"  Validation:")
            print(f"    Total: {stats['val'].get('total', 0)}")
            print(f"    Humor: {stats['val'].get('humor', 0)}")
            print(f"    Non-humor: {stats['val'].get('non_humor', 0)}")
    
    return output_files

def main():
    """Main function to merge humor manifests"""
    parser = argparse.ArgumentParser(description="Merge humor dataset manifests")
    parser.add_argument("--output_prefix", type=str, default="combined", 
                        help="Prefix for output files")
    parser.add_argument("--balance", action="store_true", default=True,
                        help="Balance positive and negative examples")
    args = parser.parse_args()
    
    setup_directories()
    
    # Find available humor manifests
    ur_funny_train = "datasets/manifests/humor/ur_funny_train_humor.csv"
    ur_funny_val = "datasets/manifests/humor/ur_funny_val_humor.csv"
    
    short_humor_train = "datasets/manifests/humor/short_humor_train_humor.csv"
    short_humor_val = "datasets/manifests/humor/short_humor_val_humor.csv"
    
    manifests = []
    
    # Add UR-FUNNY if available
    if os.path.exists(ur_funny_train) or os.path.exists(ur_funny_val):
        manifests.append({
            'name': 'UR-FUNNY',
            'train': ur_funny_train if os.path.exists(ur_funny_train) else None,
            'val': ur_funny_val if os.path.exists(ur_funny_val) else None
        })
    
    # Add Short-Humor if available
    if os.path.exists(short_humor_train) or os.path.exists(short_humor_val):
        manifests.append({
            'name': 'Short-Humor',
            'train': short_humor_train if os.path.exists(short_humor_train) else None,
            'val': short_humor_val if os.path.exists(short_humor_val) else None
        })
    
    if not manifests:
        print("No humor manifests found. Run download_ur_funny.py or download_short_humor.py first.")
        return
    
    output_files = merge_humor_manifests(manifests, args.output_prefix, args.balance)
    
    print("\nHumor manifests merged successfully!")
    print("\nYou can now train a model using the combined manifests:")
    print(f"  python enhanced_train_distil_humor.py --train_manifest {output_files.get('train', '')} --val_manifest {output_files.get('val', '')}")

if __name__ == "__main__":
    main()
