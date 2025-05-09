#!/usr/bin/env python3
import pandas as pd
import os
import argparse

def fix_dataset(file_path, output_path=None):
    """
    Fix dataset with missing or empty transcripts.
    
    Args:
        file_path: Path to the dataset CSV
        output_path: Path to save the fixed dataset. If None, overwrites the original.
    """
    print(f"Processing {file_path}")
    
    # Read the dataset
    df = pd.read_csv(file_path)
    
    # Print dataset information
    total = len(df)
    label_counts = df['label'].value_counts().to_dict()
    
    print(f"Total records: {total}")
    print(f"Label distribution: {label_counts}")
    
    # Check for empty transcripts
    empty_transcripts = df['transcript'].isna().sum()
    empty_transcripts += (df['transcript'] == '').sum()
    
    print(f"Records with empty transcripts: {empty_transcripts}")
    
    # Show label distribution for empty transcripts
    if empty_transcripts > 0:
        empty_df = df[df['transcript'].isna() | (df['transcript'] == '')]
        empty_label_counts = empty_df['label'].value_counts().to_dict()
        print(f"Label distribution for empty transcripts: {empty_label_counts}")
    
    # Keep only records with non-empty transcripts
    print("\nFixing dataset by removing rows with empty transcripts...")
    original_count = len(df)
    df = df[df['transcript'].notna() & (df['transcript'] != '')]
    new_count = len(df)
    
    print(f"Removed {original_count - new_count} records")
    print(f"New label distribution: {df['label'].value_counts().to_dict()}")
    
    # Save the fixed dataset
    if output_path is None:
        output_path = file_path
    
    df.to_csv(output_path, index=False)
    print(f"Saved fixed dataset to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix humor dataset with missing transcripts")
    parser.add_argument("--train", default="datasets/manifests/humor/train_humor_with_text.csv", 
                        help="Path to training dataset")
    parser.add_argument("--val", default="datasets/manifests/humor/val_humor_with_text.csv",
                        help="Path to validation dataset")
    parser.add_argument("--output_dir", default=None, 
                        help="Directory to save fixed datasets. If not provided, original files will be overwritten.")
    
    args = parser.parse_args()
    
    train_output = os.path.join(args.output_dir, os.path.basename(args.train)) if args.output_dir else None
    val_output = os.path.join(args.output_dir, os.path.basename(args.val)) if args.output_dir else None
    
    fix_dataset(args.train, train_output)
    print("\n" + "-"*50 + "\n")
    fix_dataset(args.val, val_output)
