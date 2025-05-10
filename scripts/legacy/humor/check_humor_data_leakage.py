#!/usr/bin/env python3
"""
Script to check for data leakage between humor train and validation sets.
This identifies cases where text from the same talk_id or with identical content
appears in both training and validation sets, which would artificially inflate
validation accuracy.
"""

import pandas as pd
import hashlib
import os
import argparse
from collections import defaultdict

def hash_text(text):
    """Create a hash of the text for exact duplicate detection"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def check_leakage(train_manifest, val_manifest):
    """Check for data leakage between training and validation sets"""
    print(f"Analyzing train manifest: {train_manifest}")
    print(f"Analyzing validation manifest: {val_manifest}")
    
    # Load manifests
    try:
        train_df = pd.read_csv(train_manifest)
        val_df = pd.read_csv(val_manifest)
    except Exception as e:
        print(f"Error loading manifests: {e}")
        return
    
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Check if we have a source identifier column
    source_id_column = None
    for col in ['talk_id', 'source', 'rel_audio', 'rel_video']:
        if col in train_df.columns and col in val_df.columns:
            source_id_column = col
            break
    
    has_source_id = source_id_column is not None
    
    # Check for source_id overlap (source-level leakage)
    if has_source_id:
        train_source_ids = set(train_df[source_id_column])
        val_source_ids = set(val_df[source_id_column])
        overlapping_source_ids = train_source_ids.intersection(val_source_ids)
        
        source_leakage_percent = len(overlapping_source_ids) / len(val_source_ids) * 100 if val_source_ids else 0
        
        print(f"\n=== Source ID Leakage (using '{source_id_column}') ===")
        print(f"Unique source IDs in train: {len(train_source_ids)}")
        print(f"Unique source IDs in val: {len(val_source_ids)}")
        print(f"Overlapping source IDs: {len(overlapping_source_ids)} ({source_leakage_percent:.2f}% of validation)")
        
        # Show some examples of overlapping source_ids
        if overlapping_source_ids:
            print("\nSample of overlapping source IDs:")
            for source_id in list(overlapping_source_ids)[:5]:  # Show up to 5 examples
                print(f"  {source_id}")
    
    # Check for exact text duplicates (content-level leakage)
    print(f"\n=== Content Leakage ===")
    
    # Determine which column contains the text
    text_column = None
    for col in ['text', 'transcript']:
        if col in train_df.columns and col in val_df.columns:
            text_column = col
            break
    
    if not text_column:
        print("No text column found in the manifests.")
        return
    
    print(f"Using '{text_column}' as the text column")
    
    # Create text hashes
    train_df['text_hash'] = train_df[text_column].apply(hash_text)
    val_df['text_hash'] = val_df[text_column].apply(hash_text)
    
    # Find duplicate texts
    train_hashes = set(train_df['text_hash'])
    val_hashes = set(val_df['text_hash'])
    overlapping_hashes = train_hashes.intersection(val_hashes)
    
    content_leakage_percent = len(overlapping_hashes) / len(val_hashes) * 100 if val_hashes else 0
    
    print(f"Unique text hashes in train: {len(train_hashes)}")
    print(f"Unique text hashes in val: {len(val_hashes)}")
    print(f"Exact duplicate texts: {len(overlapping_hashes)} ({content_leakage_percent:.2f}% of validation)")
    
    # Show some examples of duplicate content
    if overlapping_hashes:
        print("\nSample of duplicate texts:")
        sample_hashes = list(overlapping_hashes)[:3]  # Show up to 3 examples
        
        for text_hash in sample_hashes:
            train_example = train_df[train_df['text_hash'] == text_hash].iloc[0]
            val_example = val_df[val_df['text_hash'] == text_hash].iloc[0]
            
            print(f"\nHash: {text_hash}")
            print(f"Train {text_column}: {train_example[text_column][:100]}{'...' if len(train_example[text_column]) > 100 else ''}")
            print(f"Val {text_column}: {val_example[text_column][:100]}{'...' if len(val_example[text_column]) > 100 else ''}")
            if has_source_id:
                print(f"Train {source_id_column}: {train_example[source_id_column]}")
                print(f"Val {source_id_column}: {val_example[source_id_column]}")
    
    # Additional analysis: Label consistency for overlapping samples
    if overlapping_hashes and 'label' in train_df.columns and 'label' in val_df.columns:
        train_hash_to_label = train_df.set_index('text_hash')['label'].to_dict()
        val_hash_to_label = val_df.set_index('text_hash')['label'].to_dict()
        
        consistent_labels = sum(1 for h in overlapping_hashes if train_hash_to_label[h] == val_hash_to_label[h])
        
        consistency_percent = consistent_labels / len(overlapping_hashes) * 100
        
        print(f"\n=== Label Consistency ===")
        print(f"Duplicate samples with consistent labels: {consistent_labels}/{len(overlapping_hashes)} ({consistency_percent:.2f}%)")
    
    # Compute overall leakage score
    print(f"\n=== Overall Leakage Assessment ===")
    leakage_score = content_leakage_percent  # Base on content leakage
    
    if leakage_score < 1:
        print("✅ Low leakage: <1% overlap between train and validation sets")
    elif leakage_score < 5:
        print("⚠️ Moderate leakage: 1-5% overlap - should be improved for reliable evaluation")
    elif leakage_score < 20:
        print("🚨 High leakage: 5-20% overlap - evaluation metrics are compromised")
    else:
        print("🔥 Critical leakage: >20% overlap - validation metrics are meaningless")
    
    return (has_source_id, overlapping_source_ids if has_source_id else None, 
            train_hashes, val_hashes, overlapping_hashes)

def main():
    parser = argparse.ArgumentParser(description='Check for data leakage in humor datasets')
    parser.add_argument('--train', default='datasets/manifests/humor/train_humor_with_text.csv',
                        help='Path to training manifest')
    parser.add_argument('--val', default='datasets/manifests/humor/val_humor_with_text.csv',
                        help='Path to validation manifest')
    parser.add_argument('--also-check-combined', action='store_true',
                        help='Also check combined manifests')
    
    args = parser.parse_args()
    
    # Check original manifests
    check_leakage(args.train, args.val)
    
    # Check combined manifests if requested
    if args.also_check_combined:
        if os.path.exists('datasets/manifests/humor/combined_train_humor.csv') and \
           os.path.exists('datasets/manifests/humor/combined_val_humor.csv'):
            print("\n" + "="*70)
            print("Checking combined manifests...")
            check_leakage('datasets/manifests/humor/combined_train_humor.csv',
                         'datasets/manifests/humor/combined_val_humor.csv')
        else:
            print("\nCombined manifests not found. Skipping check.")

if __name__ == "__main__":
    main()
