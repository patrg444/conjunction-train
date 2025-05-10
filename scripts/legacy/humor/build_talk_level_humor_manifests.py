#!/usr/bin/env python3
"""
This script builds clean talk-level humor manifests from the SMILE dataset.
It fixes the data duplication and train/val overlap issues by:
1. Merging all utterances from a talk into a single transcript
2. Assigning one humor label per talk
3. Ensuring no overlap between train and validation sets
"""

import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
import random

def read_json_file(filepath):
    """Read a JSON file and return its contents."""
    with open(filepath, 'r') as f:
        return json.load(f)

def merge_talk_utterances(talk_data):
    """
    Merge all utterances from a talk into a single transcript.
    """
    merged_text = ""
    
    # Sort clips by their numerical index to maintain proper order
    clip_indices = sorted([int(idx) for idx in talk_data["Video clips"].keys()])
    
    for idx in clip_indices:
        clip_data = talk_data["Video clips"].get(str(idx), {})
        utterance = clip_data.get("Utterance", "").strip()
        if utterance:
            merged_text += utterance + " "
    
    return merged_text.strip()

def extract_humor_labels(gt_humor_file, text_data):
    """
    Extract humor labels from the GT laughter reason file.
    Any talk with a laughter reason is considered humorous (label 1).
    Talks not in the GT file are considered non-humorous (label 0).
    """
    humor_data = read_json_file(gt_humor_file)
    humor_labels = {}
    
    # Mark all talks with laughter reasons as humorous (1)
    for talk_id in humor_data:
        humor_labels[talk_id] = 1
    
    # Mark all other talks as non-humorous (0)
    for talk_id in text_data:
        if talk_id not in humor_labels:
            humor_labels[talk_id] = 0
        
    return humor_labels

def get_train_val_split(split_file):
    """Extract train/val split information from the data split file."""
    split_data = read_json_file(split_file)
    train_ids = []
    val_ids = []
    
    for split_info in split_data:
        for key, ids in split_info.items():
            if key.startswith('train'):
                train_ids.extend(ids)
            elif key.startswith('val') or key.startswith('test'):
                val_ids.extend(ids)
    
    return set(train_ids), set(val_ids)

def build_manifests(data_dir, output_dir):
    """
    Build talk-level humor manifests from the SMILE dataset.
    """
    # Define file paths
    text_file = os.path.join(data_dir, "annotations", "multimodal_textual_representation.json")
    split_file = os.path.join(data_dir, "annotations", "data_split.json")
    humor_file = os.path.join(data_dir, "annotations", "GT_laughter_reason.json")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read files
    print("Reading dataset files...")
    text_data = read_json_file(text_file)
    train_ids, val_ids = get_train_val_split(split_file)
    humor_labels = extract_humor_labels(humor_file, text_data)
    
    # Prepare data
    print("Processing talk transcripts...")
    talk_texts = {}
    for talk_id, talk_data in tqdm(text_data.items()):
        talk_texts[talk_id] = {
            'text': merge_talk_utterances(talk_data),
            'title': talk_data.get("Video title", ""),
            'label': humor_labels.get(talk_id, 0)  # Default to 0 if not in humor file
        }
    
    # Build train and validation manifests
    train_data = []
    val_data = []
    
    print("Building train/val manifests...")
    for talk_id, talk_info in talk_texts.items():
        # Skip talks with empty transcripts
        if not talk_info['text']:
            continue
            
        entry = {
            'talk_id': talk_id,
            'title': talk_info['title'],
            'transcript': talk_info['text'],
            'label': talk_info['label']
        }
        
        if talk_id in train_ids:
            train_data.append(entry)
        elif talk_id in val_ids:
            val_data.append(entry)
        else:
            # If ID not in split file, randomly assign to train or val
            if random.random() < 0.8:  # 80% to train, 20% to val
                train_data.append(entry)
            else:
                val_data.append(entry)
    
    # Convert to pandas DataFrames
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    
    # Check for class imbalance
    print("\nClass distribution:")
    print("Training set:")
    train_label_counts = train_df['label'].value_counts()
    print(f"  Label 0 (non-humorous): {train_label_counts.get(0, 0)} ({train_label_counts.get(0, 0)/len(train_df)*100:.1f}%)")
    print(f"  Label 1 (humorous): {train_label_counts.get(1, 0)} ({train_label_counts.get(1, 0)/len(train_df)*100:.1f}%)")
    
    print("Validation set:")
    val_label_counts = val_df['label'].value_counts()
    print(f"  Label 0 (non-humorous): {val_label_counts.get(0, 0)} ({val_label_counts.get(0, 0)/len(val_df)*100:.1f}%)")
    print(f"  Label 1 (humorous): {val_label_counts.get(1, 0)} ({val_label_counts.get(1, 0)/len(val_df)*100:.1f}%)")
    
    # Check for text overlap
    train_texts = set(train_df['transcript'])
    val_texts = set(val_df['transcript'])
    overlap = train_texts.intersection(val_texts)
    
    if overlap:
        print(f"\nWARNING: Found {len(overlap)} overlapping transcripts between train and validation!")
        print("Removing overlapping transcripts from validation set...")
        
        # Remove overlapping samples from validation
        val_df = val_df[~val_df['transcript'].isin(overlap)]
        
        # Re-check class distribution
        print("\nUpdated validation set class distribution:")
        val_label_counts = val_df['label'].value_counts()
        print(f"  Label 0 (non-humorous): {val_label_counts.get(0, 0)}")
        print(f"  Label 1 (humorous): {val_label_counts.get(1, 0)}")
    else:
        print("\nSuccess: No transcript overlap between train and validation sets")
    
    # Save to CSV
    train_output = os.path.join(output_dir, "talk_level_train_humor.csv")
    val_output = os.path.join(output_dir, "talk_level_val_humor.csv")
    
    train_df.to_csv(train_output, index=False)
    val_df.to_csv(val_output, index=False)
    
    print(f"\nSaved manifests to:")
    print(f"  Training set: {train_output} ({len(train_df)} talks)")
    print(f"  Validation set: {val_output} ({len(val_df)} talks)")
    
    return train_output, val_output

def add_text_to_manifests(train_path, val_path):
    """
    Update the humor manifests to have the column name 'text' instead of 'transcript'
    for compatibility with the existing dataloaders.
    """
    for path in [train_path, val_path]:
        df = pd.read_csv(path)
        if 'transcript' in df.columns and 'text' not in df.columns:
            df = df.rename(columns={'transcript': 'text'})
            df.to_csv(path, index=False)
    
    print("Updated manifests with 'text' column name for dataloader compatibility")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build talk-level humor manifests from SMILE dataset")
    parser.add_argument("--data-dir", default="/home/ubuntu/datasets/SMILE/raw/SMILE_DATASET",
                       help="Path to the SMILE dataset directory")
    parser.add_argument("--output-dir", default="/home/ubuntu/conjunction-train/datasets/manifests/humor",
                       help="Directory to save the manifests")
    
    args = parser.parse_args()
    
    train_path, val_path = build_manifests(args.data_dir, args.output_dir)
    add_text_to_manifests(train_path, val_path)
