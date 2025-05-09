#!/usr/bin/env python3
import pandas as pd
import os

import argparse # Import argparse

# --- Configuration ---
# Removed hardcoded path
# --- End Configuration ---

print("--- Starting Humor Dataset Class Balance Check ---")

try:
    # Use argparse to get the manifest path from command line arguments
    parser = argparse.ArgumentParser(description="Check class balance in a humor dataset manifest.")
    parser.add_argument('--manifest_path', required=True, help='Path to the humor dataset manifest CSV file.')
    args = parser.parse_args()
    manifest_path = args.manifest_path # Assign argument to manifest_path

    # Read the manifest file
    humor_df = pd.read_csv(manifest_path) # Use the argument path

    # Ensure 'split' and 'label' columns exist
    if 'split' not in humor_df.columns:
        print(f"Error: Missing 'split' column in {manifest_path}") # Use manifest_path
        exit(1)
    if 'label' not in humor_df.columns:
        print(f"Error: Missing 'label' column in {manifest_path}") # Use manifest_path
        exit(1)

    # Get data for train and validation splits
    train_df = humor_df[humor_df['split'] == 'train']
    val_df = humor_df[humor_df['split'] == 'val']

    print("\n--- Class Distribution in Training Set ---")
    if not train_df.empty:
        train_label_counts = train_df['label'].value_counts()
        print(train_label_counts)
    else:
        print("Training set is empty.")

    print("\n--- Class Distribution in Validation Set ---")
    if not val_df.empty:
        val_label_counts = val_df['label'].value_counts()
        print(val_label_counts)
    else:
        print("Validation set is empty.")

except FileNotFoundError:
    print(f"Error: Manifest file not found at {humor_manifest_path}")
    exit(1)
except Exception as e:
    print(f"An error occurred: {e}")
    exit(1)

    print(f"\n--- Humor Dataset Class Balance Check Complete for {manifest_path} ---")

# Removed incorrect main() call
# if __name__ == "__main__":
#     main()
