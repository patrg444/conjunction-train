#!/usr/bin/env python3
"""
Relabels HuBERT NPZ embedding files based on a master manifest CSV.

Reads each specified NPZ file (containing 'embeddings' and 'labels' arrays)
and its corresponding split CSV file (containing original file paths).
Looks up the basename of each file path in the master manifest to get the
correct label index and writes a new NPZ file (or overwrites the original)
with the original embeddings and the remapped labels.
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import shutil

# Ensure consistency with training script
EMOTION_LABELS = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad']

def main():
    parser = argparse.ArgumentParser(description="Relabel HuBERT NPZ files based on a master manifest.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to the master manifest CSV (e.g., combined_manifest_all.csv)")
    parser.add_argument("--emb_dir", type=str, required=True, help="Directory containing the Hubert NPZ embedding files")
    parser.add_argument("--splits_dir", type=str, required=True, help="Directory containing the original train/val/test CSV files used for embedding generation")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite original NPZ files instead of creating '_relabeled.npz'. Originals are backed up to '.bak'.")
    parser.add_argument("--path_column", type=str, default="path", help="Column name in split CSVs containing the file paths.")

    args = parser.parse_args()

    # --- 1. Load Master Manifest and Create Lookup ---
    print(f"Loading master manifest: {args.manifest}")
    try:
        manifest_df = pd.read_csv(args.manifest)
    except FileNotFoundError:
        print(f"Error: Master manifest file not found at {args.manifest}")
        return
    except Exception as e:
        print(f"Error loading master manifest: {e}")
        return

    # Ensure required columns exist
    if 'label' not in manifest_df.columns or 'path' not in manifest_df.columns:
         print("Error: Master manifest must contain 'path' and 'label' columns.")
         return

    # Create label mapping
    label_to_idx = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
    manifest_df['manifest_label_idx'] = manifest_df['label'].map(label_to_idx)

    # Create basename -> label_idx lookup
    basename_to_label = {}
    for _, row in manifest_df.iterrows():
        basename = Path(row['path']).stem
        if pd.notna(row['manifest_label_idx']): # Only add if label is valid
             basename_to_label[basename] = int(row['manifest_label_idx'])
        # else: # Optional: Warn about rows with labels not in EMOTION_LABELS
             # print(f"Warning: Basename '{basename}' has invalid label '{row['label']}' in manifest. Skipping.")

    print(f"Created lookup map for {len(basename_to_label)} basenames from manifest.")
    if not basename_to_label:
        print("Error: No valid basenames found in the manifest. Cannot proceed.")
        return

    # --- 2. Define NPZ/CSV Pairs based on existing *_regen.npz files ---
    print(f"Scanning {args.emb_dir} for *_regen.npz files...")
    npz_files = list(Path(args.emb_dir).glob('*_regen.npz'))
    if not npz_files:
        print(f"Error: No '*_regen.npz' files found in {args.emb_dir}")
        return

    embedding_sources = []
    for npz_path in npz_files:
        # Infer corresponding CSV name (e.g., train_embeddings_regen.npz -> train.csv)
        base_name = npz_path.stem.replace('_embeddings_regen', '')
        csv_name = f"{base_name}.csv"
        embedding_sources.append({'npz': npz_path.name, 'csv': csv_name})
        print(f"  Found: {npz_path.name} -> Will use CSV: {csv_name}")

    # --- 3. Process Each Found NPZ/CSV Pair ---
    total_files_processed = 0
    total_labels_remapped = 0
    total_labels_missing_manifest = 0

    for source in embedding_sources:
        npz_path = Path(args.emb_dir) / source['npz']
        csv_path = Path(args.splits_dir) / source['csv']

        if not npz_path.exists():
            # print(f"Info: Skipping NPZ file (not found): {npz_path}")
            continue
        if not csv_path.exists():
            print(f"Warning: Skipping NPZ file {npz_path.name} because corresponding CSV not found: {csv_path}")
            continue

        print(f"\nProcessing NPZ: {npz_path.name} using CSV: {csv_path.name}")
        total_files_processed += 1

        try:
            # Load NPZ data
            hubert_data = np.load(npz_path)
            embeddings = hubert_data['embeddings']
            old_labels = hubert_data['labels']
            print(f"  Loaded {len(embeddings)} embeddings and {len(old_labels)} old labels.")

            # Load corresponding split CSV
            split_df = pd.read_csv(csv_path)
            if args.path_column not in split_df.columns:
                 print(f"  Error: Path column '{args.path_column}' not found in {csv_path.name}. Skipping this file.")
                 continue
            if len(split_df) != len(embeddings):
                 print(f"  Error: Mismatch between CSV rows ({len(split_df)}) and embeddings ({len(embeddings)}). Skipping this file.")
                 continue

            # Remap labels
            new_labels = np.zeros_like(old_labels)
            remapped_count = 0
            missing_count = 0

            for i, row in tqdm(split_df.iterrows(), total=len(split_df), desc="  Remapping labels", unit="sample"):
                original_path = row[args.path_column]
                basename = Path(original_path).stem
                manifest_label_idx = basename_to_label.get(basename)

                if manifest_label_idx is not None:
                    new_labels[i] = manifest_label_idx
                    if old_labels[i] != new_labels[i]:
                        remapped_count += 1
                else:
                    # Keep original label if basename not in manifest
                    new_labels[i] = old_labels[i]
                    missing_count += 1
                    # Optional: More verbose warning
                    # print(f"  Warning: Basename '{basename}' (from {original_path}) not found in master manifest. Keeping original label {old_labels[i]}.")

            print(f"  Remapping complete for {npz_path.name}:")
            print(f"    Labels remapped to match manifest: {remapped_count}")
            print(f"    Labels kept original (basename not in manifest): {missing_count}")
            total_labels_remapped += remapped_count
            total_labels_missing_manifest += missing_count

            # Determine output path and save
            if args.overwrite:
                output_npz_path = npz_path
                backup_path = npz_path.with_suffix(npz_path.suffix + '.bak')
                print(f"  Overwriting original file. Backing up to: {backup_path.name}")
                try:
                    shutil.copy2(npz_path, backup_path) # copy2 preserves metadata
                except Exception as backup_err:
                    print(f"  Error creating backup: {backup_err}. Aborting save for this file.")
                    continue
            else:
                output_npz_path = npz_path.with_name(f"{npz_path.stem}_relabeled.npz")
                print(f"  Saving remapped data to: {output_npz_path.name}")

            try:
                np.savez_compressed(output_npz_path, embeddings=embeddings, labels=new_labels)
                print(f"  Successfully saved: {output_npz_path}")
            except Exception as save_err:
                print(f"  Error saving NPZ file: {save_err}")
                # If overwrite failed, maybe try to restore backup?
                if args.overwrite and backup_path.exists():
                    try:
                        print(f"  Attempting to restore backup...")
                        shutil.move(backup_path, npz_path)
                        print(f"  Restored backup.")
                    except Exception as restore_err:
                        print(f"  Error restoring backup: {restore_err}. Original file might be corrupted.")


        except FileNotFoundError:
            print(f"  Error: File not found during processing: {npz_path} or {csv_path}")
        except KeyError as e:
            print(f"  Error: Missing key {e} in NPZ file {npz_path.name}. Ensure it contains 'embeddings' and 'labels'.")
        except Exception as e:
            print(f"  An unexpected error occurred processing {npz_path.name}: {e}")

    print("\n--- Overall Summary ---")
    print(f"NPZ files processed: {total_files_processed}")
    print(f"Total labels remapped: {total_labels_remapped}")
    print(f"Total labels kept original (missing in manifest): {total_labels_missing_manifest}")
    if total_labels_missing_manifest > 0:
        print("Warning: Some labels could not be remapped as their basenames were not found in the master manifest.")
    print("Relabeling process finished.")

if __name__ == "__main__":
    main()
