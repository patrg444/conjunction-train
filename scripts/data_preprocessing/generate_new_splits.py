# generate_new_splits.py
import pandas as pd
import os
from pathlib import Path

manifest_path = 'corrected_manifest.csv'
output_dir = 'splits_new'

print(f"Reading manifest: {manifest_path}")
try:
    df = pd.read_csv(manifest_path)
    print(f"Manifest loaded with {len(df)} rows.")
except Exception as e:
    print(f"Error reading manifest {manifest_path}: {e}")
    exit(1)

# Ensure the 'split' column exists
if 'split' not in df.columns:
    print("Error: 'split' column not found in manifest.")
    exit(1)

# Ensure the output directory exists
output_path = Path(output_dir)
output_path.mkdir(parents=True, exist_ok=True)
print(f"Ensured output directory exists: {output_dir}")

# Columns to keep in the split files (adjust if needed)
columns_to_keep = ['path', 'label', 'split', 'dataset', 'actor_id']
columns_to_keep = [col for col in columns_to_keep if col in df.columns] # Only keep existing columns

# Define mapping from manifest split names to output file names
split_mapping = {
    'ravdess_train': 'train',
    'crema_d_train': 'train',
    'ravdess_validation': 'val',
    'crema_d_validation': 'val',
    'ravdess_test': 'test',
    'crema_d_test': 'test'
}

# Group by the *mapped* split name and write files
split_counts = {}
# Create a temporary column for the mapped split name
df['mapped_split'] = df['split'].map(split_mapping)

# Filter out rows where the split name wasn't in the mapping
df_filtered = df.dropna(subset=['mapped_split'])
if len(df_filtered) < len(df):
    print(f"Warning: {len(df) - len(df_filtered)} rows had unexpected split values and were skipped.")

for mapped_split_name, group_df in df_filtered.groupby('mapped_split'):
    output_file = output_path / f"{mapped_split_name}.csv"
    try:
        # Select only the desired columns (original columns, not the temporary mapped one)
        group_df_to_save = group_df[columns_to_keep] # Use original columns
        group_df_to_save.to_csv(output_file, index=False)
        split_counts[mapped_split_name] = len(group_df_to_save) # Use mapped name for count key
        print(f"Successfully wrote {len(group_df_to_save)} rows to {output_file}")
    except Exception as e:
        print(f"Error writing split file {output_file}: {e}")

print("\nSplit file generation summary:")
for name, count in split_counts.items():
    print(f"  {name}.csv: {count} rows") # Use mapped name in summary

total_written = sum(split_counts.values()) # Sum counts based on mapped names
print(f"Total rows written across new splits: {total_written}")

if total_written != len(df_filtered): # Compare against the filtered dataframe length
    print(f"Warning: Total rows written ({total_written}) does not match filtered manifest rows ({len(df_filtered)}).")
elif len(df_filtered) != len(df):
     print(f"Warning: Total rows written matches filtered count, but {len(df) - len(df_filtered)} original rows were skipped due to unexpected split values.")
else:
    print("Total rows match manifest count. Split generation successful.")
