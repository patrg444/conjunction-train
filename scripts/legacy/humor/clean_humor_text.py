#!/usr/bin/env python3
"""
Clean UR-FUNNY text data by removing excess spaces between characters
to convert "t h e" format to normal "the" format.
"""

import pandas as pd
import re

def clean_text(text):
    """
    Clean the text by joining individual characters into words.
    
    This converts text like "t h e   m a t h" into "the math"
    where the 3+ spaces between "e" and "m" represent word boundaries
    """
    # Keep transcript-level separator untouched
    parts = text.split('|||')
    cleaned_parts = []
    
    for part in parts:
        # Strip edges but keep original internal spacing
        p = part.strip()
        
        # Split where we have >=2 spaces (word boundary)
        words_raw = re.split(r' {2,}', p)
        
        # Join letters inside each word (remove single spaces)
        words = [''.join(w.split()) for w in words_raw if w]
        
        # Put back together with normal spaces between words
        cleaned_parts.append(' '.join(words))
    
    # Rejoin parts with the separator
    result = ' ||| '.join(cleaned_parts)
    
    # Optional post-processing for rare edge cases
    # 1. Add space between lowercase and uppercase letters (camelCase → camel Case)
    result = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', result)
    
    # 2. Handle specific common fusions 
    result = result.replace('worksand', 'works and')
    result = result.replace('tellsus', 'tells us')
    
    return result

def process_manifest(input_file, output_file):
    """Process a manifest file to clean the text."""
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    
    print("Cleaning text...")
    df['transcript'] = df['transcript'].apply(clean_text)
    
    print(f"Writing cleaned data to {output_file}...")
    df.to_csv(output_file, index=False)
    
    # Print a sample for verification
    sample = df.iloc[0]['transcript']
    print(f"\nSample of cleaned text: {sample[:100]}...")
    
    return df

def main():
    # Process training manifest
    train_input = "datasets/manifests/humor/ur_funny_train_humor_ready.csv"
    train_output = "datasets/manifests/humor/ur_funny_train_humor_cleaned.csv"
    train_df = process_manifest(train_input, train_output)
    
    # Process validation manifest
    val_input = "datasets/manifests/humor/ur_funny_val_humor_ready.csv"
    val_output = "datasets/manifests/humor/ur_funny_val_humor_cleaned.csv"
    val_df = process_manifest(val_input, val_output)
    
    print(f"\nComplete! Processed {len(train_df)} training samples and {len(val_df)} validation samples.")

if __name__ == "__main__":
    main()
