#!/usr/bin/env python3
"""
Validate alignment between video manifest, HuBERT split CSVs, and HuBERT NPZ embeddings.
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Consistent emotion labels and mapping (must match training script)
# Aligned with ser_hubert/data_module.py LABEL_MAP
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
LABEL_TO_IDX = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
IDX_TO_LABEL = {idx: label for label, idx in LABEL_TO_IDX.items()}

def get_basename(filepath):
    """Extracts the basename without extension from a filepath string."""
    # Handles potential Windows paths as well if needed
    return Path(filepath).stem

def validate_split(manifest_df, split_csv_path, npz_path, split_name):
    """Validates data alignment for a specific split."""
    print(f"\n--- Validating Split: {split_name} ---")
    print(f"Manifest items for split (after filtering by split CSV): {len(manifest_df)}")

    # --- Load Split CSV ---
    if not split_csv_path.exists():
        print(f"ERROR: Split CSV not found: {split_csv_path}")
        return False
    try:
        split_df = pd.read_csv(split_csv_path)
    except Exception as e:
        print(f"ERROR: Failed to load split CSV {split_csv_path}: {e}")
        return False

    # Assuming the path column needs basename extraction
    path_column_in_split_csv = 'path'
    if path_column_in_split_csv not in split_df.columns:
         if 'FileName' in split_df.columns:
             path_column_in_split_csv = 'FileName'
         else:
            print(f"ERROR: Path column ('path' or 'FileName') not found in {split_csv_path}")
            return False
            
    split_df['basename'] = split_df[path_column_in_split_csv].apply(get_basename)
    split_basenames = set(split_df['basename'])
    print(f"Split CSV items ({split_csv_path.name}): {len(split_df)}")

    # --- Load NPZ ---
    if not npz_path.exists():
        print(f"ERROR: NPZ file not found: {npz_path}")
        return False # Cannot proceed without NPZ
    try:
        npz_data = np.load(npz_path)
        npz_embeddings = npz_data['embeddings']
        npz_labels = npz_data['labels']
        print(f"NPZ items ({npz_path.name}): Embeddings={len(npz_embeddings)}, Labels={len(npz_labels)}")
    except Exception as e:
        print(f"ERROR: Failed to load or read NPZ file {npz_path}: {e}")
        return False

    # --- Basic Length Checks ---
    if len(split_df) != len(npz_embeddings) or len(split_df) != len(npz_labels):
        print(f"ERROR: Length mismatch! Split CSV ({len(split_df)}) vs NPZ Embeddings ({len(npz_embeddings)}) vs NPZ Labels ({len(npz_labels)})")
        return False # Cannot reliably map if lengths differ

    # --- Compare Manifest vs Split CSV (using filtered manifest) ---
    # The manifest_df passed here is already filtered for the current split
    manifest_df['basename'] = manifest_df['path'].apply(get_basename) # Ensure basename exists
    manifest_basenames = set(manifest_df['basename'])

    # Check if all items in the filtered manifest are indeed in the split CSV's basenames
    missing_in_split_csv_after_filter = manifest_basenames - split_basenames
    if missing_in_split_csv_after_filter:
        print(f"INTERNAL WARNING: {len(missing_in_split_csv_after_filter)} items in filtered Manifest are somehow MISSING in Split CSV '{split_csv_path.name}'. This indicates a filtering logic error.")

    # Check if all items from the split CSV are present in the filtered manifest
    missing_in_manifest_after_filter = split_basenames - manifest_basenames
    if missing_in_manifest_after_filter:
         print(f"WARNING: {len(missing_in_manifest_after_filter)} items in Split CSV '{split_csv_path.name}' were NOT found in the main manifest (or had different basenames):")
         # for i, item in enumerate(missing_in_manifest_after_filter):
         #     if i < 10: print(f"  - {item}")
         # if len(missing_in_manifest_after_filter) > 10: print("  ...")
    else:
        print(f"OK: All items in Split CSV '{split_csv_path.name}' have a corresponding entry in the main manifest.")


    # --- Compare Label Distributions (Manifest vs NPZ) ---
    # Since NPZ files lack filenames, we cannot reliably map item-by-item if order differs.
    # Instead, compare the overall counts of each label index.

    # 1. Get label counts from the filtered manifest for this split
    manifest_label_counts = manifest_df['label_idx'].value_counts().sort_index()

    # 2. Get label counts from the NPZ labels array
    unique_npz_labels, npz_label_counts_array = np.unique(npz_labels, return_counts=True)
    npz_label_counts = pd.Series(npz_label_counts_array, index=unique_npz_labels).sort_index()

    # 3. Compare the counts
    labels_match = True
    print("Comparing label distributions:")
    print("  Manifest Counts:")
    for idx, count in manifest_label_counts.items():
        print(f"    - Label {idx} ({IDX_TO_LABEL.get(idx, '??')}): {count}")
    print("  NPZ Counts:")
    for idx, count in npz_label_counts.items():
        print(f"    - Label {idx} ({IDX_TO_LABEL.get(idx, '??')}): {count}")

    # Check if indices and counts match
    if not manifest_label_counts.equals(npz_label_counts):
        print(f"ERROR: Label distribution mismatch between Manifest and NPZ '{npz_path.name}'!")
        # Find differing counts for detailed reporting (optional)
        all_indices = sorted(list(set(manifest_label_counts.index) | set(npz_label_counts.index)))
        for idx in all_indices:
            m_count = manifest_label_counts.get(idx, 0)
            n_count = npz_label_counts.get(idx, 0)
            if m_count != n_count:
                print(f"  - Index {idx}: Manifest has {m_count}, NPZ has {n_count}")
        labels_match = False
    else:
        print(f"OK: Label distributions match between Manifest and NPZ '{npz_path.name}'.")

    # A split is considered valid if label distributions match *and* no length mismatches (checked earlier)
    return labels_match


def main():
    parser = argparse.ArgumentParser(description="Validate Fusion Data Alignment")
    parser.add_argument("--manifest", type=str, default="data/audio_manifest.tsv", help="Path to the main video/audio manifest file (CSV or TSV)")
    parser.add_argument("--splits_dir", type=str, default="splits", help="Directory containing train/val/test split CSVs")
    parser.add_argument("--npz_dir", type=str, default=".", help="Directory containing HuBERT NPZ embedding files")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    splits_dir = Path(args.splits_dir)
    npz_dir = Path(args.npz_dir)

    if not manifest_path.exists():
        print(f"ERROR: Main manifest file not found: {manifest_path}")
        return
    if not splits_dir.is_dir():
        print(f"ERROR: Splits directory not found: {splits_dir}")
        return
    if not npz_dir.is_dir():
        print(f"ERROR: NPZ directory not found: {npz_dir}")
        return

    # Determine separator based on file extension
    separator = '\t' if manifest_path.suffix.lower() == '.tsv' else ','
    print(f"Loading main manifest using separator '{separator}' (assuming no header)...")
    try:
        # Load with no header, let pandas assign numerical column names
        main_manifest_df = pd.read_csv(manifest_path, sep=separator, header=None)
        print(f"Initial columns loaded: {main_manifest_df.columns.tolist()}")
    except Exception as e:
        print(f"ERROR: Failed to load manifest file {manifest_path} with separator '{separator}' and no header: {e}")
        return

    # --- Attempt to identify and rename columns (assuming 0=path, 1=label, no split column) ---
    identified_path_col = None
    identified_label_col = None

    # Check if default numerical columns exist (means header=None worked)
    if 0 in main_manifest_df.columns:
        print("Manifest loaded with numerical columns (header=None). Assuming column order: 0=path, 1=label.")
        identified_path_col = 0
        if 1 in main_manifest_df.columns:
            identified_label_col = 1
    else:
         # Header might exist but with different names, try finding them
         print("Manifest might have a header with non-standard names. Attempting to identify...")
         possible_path_cols = ['path', 'file', 'filename', 'filepath', 'audio_path', 'video_path']
         possible_label_cols = ['label', 'emotion', 'category', 'class']

         for col in possible_path_cols:
             if col in main_manifest_df.columns:
                 identified_path_col = col
                 break
         for col in possible_label_cols:
             if col in main_manifest_df.columns:
                 identified_label_col = col
                 break

    # --- Validate identified columns and rename (only path and label expected now) ---
    if not all([identified_path_col is not None, identified_label_col is not None]):
        print(f"ERROR: Could not reliably identify required columns (path, label). Found/Assumed: path={identified_path_col}, label={identified_label_col}")
        print(f"Actual columns found: {main_manifest_df.columns.tolist()}")
        try:
            print("\nFirst 5 lines of manifest:")
            with open(manifest_path, 'r') as f:
                for i, line in enumerate(f):
                    if i >= 5: break
                    print(line.strip())
        except Exception as read_err:
            print(f"(Could not read first lines: {read_err})")
        return
    else:
        print(f"Using columns: path='{identified_path_col}', label='{identified_label_col}'")
        # Rename columns to standard names for downstream processing
        main_manifest_df = main_manifest_df.rename(columns={
            identified_path_col: 'path',
            identified_label_col: 'label',
        })
        # Ensure data types are correct after potential numerical loading
        main_manifest_df['path'] = main_manifest_df['path'].astype(str)
        # Attempt to convert label column directly to integer, coercing errors to NaN
        main_manifest_df['label_idx'] = pd.to_numeric(main_manifest_df['label'], errors='coerce')
        # Drop rows where label conversion failed
        original_len = len(main_manifest_df)
        main_manifest_df = main_manifest_df.dropna(subset=['label_idx'])
        if len(main_manifest_df) < original_len:
            print(f"WARNING: Dropped {original_len - len(main_manifest_df)} rows from manifest due to non-numeric labels.")
        main_manifest_df['label_idx'] = main_manifest_df['label_idx'].astype(int) # Convert valid ones to int
        # Keep original label column for reference if needed
        # main_manifest_df['original_label'] = main_manifest_df['label'] # Optional: keep original string

    print(f"Loaded and processed {len(main_manifest_df)} items from {manifest_path.name}")


    all_splits_valid = True
    # Define standard and crema-d splits to check
    # Since manifest doesn't have split info, we determine it from the split CSVs
    main_manifest_df['basename'] = main_manifest_df['path'].apply(get_basename) # Pre-calculate basenames

    splits_to_check = {
        "train": ("train.csv", "train_embeddings.npz"),
        "val": ("val.csv", "val_embeddings.npz"),
        "test": ("test.csv", "test_embeddings.npz"),
        "crema_d_train": ("crema_d_train.csv", "crema_d_train_embeddings.npz"),
        "crema_d_val": ("crema_d_val.csv", "crema_d_val_embeddings.npz"),
    }

    for split_name, (csv_filename, npz_filename) in splits_to_check.items():
        split_csv_path = splits_dir / csv_filename
        npz_path = npz_dir / npz_filename

        if not split_csv_path.exists():
            print(f"\n--- Skipping Split: {split_name} ---")
            print(f"Split CSV not found: {split_csv_path}")
            continue
        if not npz_path.exists():
             print(f"\n--- Skipping Split: {split_name} ---")
             print(f"NPZ file not found: {npz_path}")
             continue

        # Load basenames from the split CSV to filter the main manifest
        try:
            split_csv_df = pd.read_csv(split_csv_path)
            path_col_in_split = 'path' if 'path' in split_csv_df.columns else 'FileName'
            if path_col_in_split not in split_csv_df.columns:
                 print(f"ERROR: Cannot find path column in {split_csv_path}. Skipping split {split_name}.")
                 continue
            split_basenames = set(split_csv_df[path_col_in_split].apply(get_basename))
        except Exception as e:
            print(f"ERROR: Failed to load or process split CSV {split_csv_path}: {e}. Skipping split {split_name}.")
            continue

        # Filter the main manifest to only include items in this split's CSV
        manifest_split_df = main_manifest_df[main_manifest_df['basename'].isin(split_basenames)].copy()

        if manifest_split_df.empty:
            print(f"\n--- Skipping Split: {split_name} ---")
            print(f"No items from the main manifest were found in the split CSV '{csv_filename}'.")
            continue

        # Now validate this filtered manifest against the split CSV and NPZ
        if not validate_split(manifest_split_df, split_csv_path, npz_path, split_name):
            all_splits_valid = False # Mark as invalid if any check fails within the split

    print("\n--- Validation Summary ---")
    if all_splits_valid:
        print("All checked splits appear consistent in terms of labels for common items.")
        print("However, review any 'MISSING' warnings above - these indicate items present in one source but not another.")
    else:
        print("Validation FAILED: Found critical inconsistencies (label mismatches or length errors). Review errors above.")

if __name__ == "__main__":
    main()
