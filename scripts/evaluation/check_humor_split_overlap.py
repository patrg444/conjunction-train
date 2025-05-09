#!/usr/bin/env python3
import pandas as pd
import os

#!/usr/bin/env python3
import pandas as pd
import os

# --- Configuration ---
HUMOR_MANIFEST_PATH = "datasets/urfunny_manifest_textonly.csv" # Corrected path
# --- End Configuration ---

print("--- Starting Humor Dataset Split Overlap Check ---")

try:
    # Read the manifest file
    humor_df = pd.read_csv(HUMOR_MANIFEST_PATH)

    # Ensure 'split' column exists
    if 'split' not in humor_df.columns:
        print(f"Error: Missing 'split' column in {HUMOR_MANIFEST_PATH}")
        exit(1)

    # Use 'id' column as the unique identifier based on inspection
    unique_id_col = 'id'

    if unique_id_col not in humor_df.columns:
         print(f"Error: Missing expected unique ID column '{unique_id_col}' in {HUMOR_MANIFEST_PATH}")
         exit(1)

    print(f"Using column '{unique_id_col}' as the unique identifier for overlap check.")

    # Get unique identifiers for train and validation splits
    train_ids = set(humor_df[humor_df['split'] == 'train'][unique_id_col].tolist())
    val_ids = set(humor_df[humor_df['split'] == 'val'][unique_id_col].tolist())

    print(f"Found {len(train_ids)} unique train samples.")
    print(f"Found {len(val_ids)} unique validation samples.")

    # Check for overlap
    overlap_ids = train_ids.intersection(val_ids)
    num_overlap = len(overlap_ids)

    if num_overlap > 0:
        print(f"\nERROR: Found {num_overlap} overlapping samples between train and val sets!")
        # Optionally print overlapping IDs (can be large)
        # print("Overlapping sample IDs:")
        # for i, sample_id in enumerate(overlap_ids):
        #     print(f"  {i+1}. {sample_id}")
    else:
        print("\nSuccess: No overlap found between training and validation sets.")

except FileNotFoundError:
    print(f"Error: Manifest file not found at {HUMOR_MANIFEST_PATH}")
    exit(1)
except Exception as e:
    print(f"An error occurred: {e}")
    exit(1)

print("--- Humor Dataset Split Overlap Check Complete ---")
