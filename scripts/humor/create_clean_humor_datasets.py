#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import argparse
from collections import Counter
from sklearn.model_selection import train_test_split

def clean_dataset(df):
    """
    Clean the dataset by removing duplicates and standardizing text
    """
    print(f"Original dataset size: {len(df)}")
    
    # Remove exact duplicates
    df_unique = df.drop_duplicates(subset=['transcript'])
    print(f"After removing exact duplicates: {len(df_unique)}")
    
    # Check if there are any rows with empty transcripts
    empty_transcripts = df_unique['transcript'].isna().sum()
    if empty_transcripts > 0:
        print(f"Removing {empty_transcripts} rows with empty transcripts")
        df_unique = df_unique.dropna(subset=['transcript'])
    
    # Check if all rows end with "(audience laughs)" and remove this suffix if needed
    pattern = " (audience laughs)"
    has_pattern = df_unique['transcript'].str.endswith(pattern).mean() > 0.9
    
    if has_pattern:
        print(f"Removing common ending pattern: '{pattern}'")
        df_unique['transcript'] = df_unique['transcript'].str.replace(pattern + '$', '', regex=True)
        
        # Remove duplicates again after pattern removal
        old_count = len(df_unique)
        df_unique = df_unique.drop_duplicates(subset=['transcript'])
        print(f"After removing duplicates with standardized ending: {len(df_unique)} (removed {old_count - len(df_unique)})")
    
    return df_unique

def check_for_phrase_bias(df, threshold=0.1):
    """
    Check for phrases that appear often and are highly correlated with specific labels
    """
    # Count the most common starting sequences (first 20 chars)
    df['start_seq'] = df['transcript'].apply(lambda x: x[:20] if len(x) >= 20 else x)
    start_counts = Counter(df['start_seq'])
    
    # Identify phrases that appear frequently
    common_phrases = [phrase for phrase, count in start_counts.items() 
                     if count / len(df) > threshold]
    
    if common_phrases:
        print(f"\nFound {len(common_phrases)} common starting phrases (appearing in >{threshold*100}% of data):")
        for phrase in common_phrases:
            # Calculate correlation with label
            has_phrase = df['start_seq'] == phrase
            label_with_phrase = df.loc[has_phrase, 'label'].mean()
            phrase_count = has_phrase.sum()
            percent = phrase_count / len(df) * 100
            
            print(f"  '{phrase}...' appears in {phrase_count} samples ({percent:.1f}%), label correlation: {label_with_phrase:.2f}")
            
            # If a phrase appears in >25% of data with strong label correlation, warn about it
            if percent > 25 and (label_with_phrase > 0.9 or label_with_phrase < 0.1):
                print(f"  ⚠️ WARNING: This phrase strongly correlates with {'positive' if label_with_phrase > 0.5 else 'negative'} class ({label_with_phrase:.2f})")
                # Consider downsampling this phrase
                if percent > 40:
                    print(f"  ⚠️ RECOMMENDATION: This phrase dominates the dataset and should be downsampled")
    
    return common_phrases

def balance_dataset(df, common_phrases=None, max_phrase_percent=0.2):
    """
    Balance the dataset by:
    1. Ensuring class balance (equal positive/negative)
    2. Limiting over-represented starting phrases
    """
    print("\nBalancing dataset...")
    
    # First, ensure no single phrase dominates the dataset
    if common_phrases:
        for phrase in common_phrases:
            mask = df['transcript'].str.startswith(phrase)
            count = mask.sum()
            max_allowed = int(len(df) * max_phrase_percent)
            
            if count > max_allowed:
                to_remove = count - max_allowed
                print(f"Downsampling phrase '{phrase}...' from {count} to {max_allowed} samples")
                
                # Get indices of samples with this phrase
                phrase_indices = df[mask].index
                
                # Randomly select indices to remove
                remove_indices = np.random.choice(phrase_indices, size=to_remove, replace=False)
                df = df.drop(remove_indices)
    
    # Now balance positive and negative classes
    neg_samples = df[df['label'] == 0]
    pos_samples = df[df['label'] == 1]
    
    print(f"Class distribution before balancing: {len(neg_samples)} negative, {len(pos_samples)} positive")
    
    # Downsample the majority class
    if len(neg_samples) > len(pos_samples):
        neg_samples = neg_samples.sample(len(pos_samples), random_state=42)
    else:
        pos_samples = pos_samples.sample(len(neg_samples), random_state=42)
    
    # Combine balanced samples
    df_balanced = pd.concat([neg_samples, pos_samples])
    
    print(f"Final balanced dataset size: {len(df_balanced)} samples " 
          f"({len(df_balanced[df_balanced['label'] == 0])} negative, "
          f"{len(df_balanced[df_balanced['label'] == 1])} positive)")
    
    return df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

def main():
    parser = argparse.ArgumentParser(description="Clean and balance humor datasets")
    parser.add_argument("--train", default="datasets/manifests/humor/train_humor_with_text.csv", 
                        help="Path to training dataset")
    parser.add_argument("--val", default="datasets/manifests/humor/val_humor_with_text.csv",
                        help="Path to validation dataset")
    parser.add_argument("--output_dir", default="datasets/manifests/humor/", 
                        help="Path to output directory")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Proportion of data to use for validation")
    
    args = parser.parse_args()
    
    # Make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read and combine datasets
    train_df = pd.read_csv(args.train)
    val_df = pd.read_csv(args.val)
    print(f"Read {len(train_df)} training samples and {len(val_df)} validation samples")
    
    # Combine datasets for cleaning
    combined_df = pd.concat([train_df, val_df])
    print(f"Combined dataset size: {len(combined_df)}")
    
    # Clean dataset (remove duplicates, standardize text)
    cleaned_df = clean_dataset(combined_df)
    
    # Check for phrase bias
    common_phrases = check_for_phrase_bias(cleaned_df, threshold=0.1)
    
    # Balance dataset
    balanced_df = balance_dataset(cleaned_df, common_phrases)
    
    # Create new train/validation split
    train_df_new, val_df_new = train_test_split(
        balanced_df, test_size=args.test_size, stratify=balanced_df['label'], 
        random_state=42
    )
    
    print(f"\nNew train/validation split:")
    print(f"Training set: {len(train_df_new)} samples "
          f"({len(train_df_new[train_df_new['label'] == 0])} negative, "
          f"{len(train_df_new[train_df_new['label'] == 1])} positive)")
    print(f"Validation set: {len(val_df_new)} samples "
          f"({len(val_df_new[val_df_new['label'] == 0])} negative, "
          f"{len(val_df_new[val_df_new['label'] == 1])} positive)")
    
    # Save new datasets
    train_output = os.path.join(args.output_dir, "clean_train_humor_with_text.csv")
    val_output = os.path.join(args.output_dir, "clean_val_humor_with_text.csv")
    
    train_df_new.to_csv(train_output, index=False)
    val_df_new.to_csv(val_output, index=False)
    
    print(f"\nSaved cleaned datasets to:")
    print(f"Training set: {train_output}")
    print(f"Validation set: {val_output}")
    
    # Check for overlap between new train and validation sets
    train_texts = set(train_df_new['transcript'])
    val_texts = set(val_df_new['transcript'])
    overlap = train_texts.intersection(val_texts)
    
    if overlap:
        print(f"\nWARNING: Found {len(overlap)} overlapping samples between train and validation!")
    else:
        print("\nSuccess: No text overlap between train and validation sets")

if __name__ == "__main__":
    main()
