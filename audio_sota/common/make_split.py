import pandas as pd
import argparse
from pathlib import Path
# from sklearn.model_selection import GroupShuffleSplit # Replaced with manual assignment for exact counts
import numpy as np # For random shuffling

def create_splits_by_count(metadata_csv, output_dir, n_test_speakers, n_val_speakers, random_state=42):
    """Creates speaker-independent train/validation/test splits based on speaker counts."""
    metadata_path = Path(metadata_csv)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not metadata_path.exists():
        print(f"Error: Metadata file not found at {metadata_path}")
        return
        
    df = pd.read_csv(metadata_path)
    
    print(f"Loaded metadata with {len(df)} samples.")
    
    # Ensure speaker column is suitable for grouping (e.g., convert to string if needed)
    df['speaker'] = df['speaker'].astype(str)
    all_speakers = df['speaker'].unique()
    n_total_speakers = len(all_speakers)
    print(f"Total unique speakers found: {n_total_speakers}")

    if n_test_speakers + n_val_speakers >= n_total_speakers:
        raise ValueError(f"Sum of test speakers ({n_test_speakers}) and validation speakers ({n_val_speakers}) "
                         f"must be less than the total number of speakers ({n_total_speakers}).")

    # --- Manual Speaker Assignment ---
    np.random.seed(random_state)
    shuffled_speakers = np.random.permutation(all_speakers)

    test_speakers = set(shuffled_speakers[:n_test_speakers])
    val_speakers = set(shuffled_speakers[n_test_speakers : n_test_speakers + n_val_speakers])
    train_speakers = set(shuffled_speakers[n_test_speakers + n_val_speakers:])

    n_train_speakers = len(train_speakers)
    print(f"\nAssigning speakers to splits:")
    print(f"  Test speakers ({len(test_speakers)}): {sorted(list(test_speakers))}")
    print(f"  Validation speakers ({len(val_speakers)}): {sorted(list(val_speakers))}")
    print(f"  Train speakers ({len(train_speakers)}): {sorted(list(train_speakers))}") # Print first few if too many?

    # --- Assign split labels and save ---
    df['split'] = 'unknown' # Default
    df.loc[df['speaker'].isin(train_speakers), 'split'] = 'train'
    df.loc[df['speaker'].isin(val_speakers), 'split'] = 'validation'
    df.loc[df['speaker'].isin(test_speakers), 'split'] = 'test'

    # Save individual split files
    train_df_out = df[df['split'] == 'train']
    val_df_out = df[df['split'] == 'validation']
    test_df_out = df[df['split'] == 'test']
    train_csv = output_path / 'train.csv'
    val_csv = output_path / 'validation.csv' # Keep consistent naming
    test_csv = output_path / 'test.csv'
    
    train_df_out.to_csv(train_csv, index=False)
    val_df_out.to_csv(val_csv, index=False)
    test_df_out.to_csv(test_csv, index=False)
    
    print(f"\nSplit files saved:")
    print(f"  Train: {train_csv} ({len(train_df_out)} samples, {len(train_speakers)} speakers)")
    print(f"  Validation: {val_csv} ({len(val_df_out)} samples, {len(val_speakers)} speakers)")
    print(f"  Test: {test_csv} ({len(test_df_out)} samples, {len(test_speakers)} speakers)")

    # Verification (already done by construction with sets)
    assert train_speakers.isdisjoint(val_speakers), "Speaker overlap detected between train and validation!"
    assert train_speakers.isdisjoint(test_speakers), "Speaker overlap detected between train and test!"
    assert val_speakers.isdisjoint(test_speakers), "Speaker overlap detected between validation and test!"
    print("\nSpeaker disjointness verified between splits.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create speaker-independent train/val/test splits based on speaker counts.")
    parser.add_argument('--csv', required=True, help='Path to the input metadata.csv file.')
    parser.add_argument('--output_dir', required=True, help='Path to the directory to save the split CSV files (e.g., audio_sota/data/dataset_name/splits).')
    parser.add_argument('--test_speakers', type=int, required=True, help='Number of unique speakers for the test set.')
    parser.add_argument('--val_speakers', type=int, required=True, help='Number of unique speakers for the validation set.')
    parser.add_argument('--seed', type=int, default=42, help='Random state for reproducibility.')
    
    args = parser.parse_args()
    
    create_splits_by_count(args.csv, args.output_dir, 
                           n_test_speakers=args.test_speakers, 
                           n_val_speakers=args.val_speakers, 
                           random_state=args.seed)
