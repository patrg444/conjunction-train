import pandas as pd
import os
import sys

# Define the source and target directories
source_dir = "splits"
target_dir = "splits" # Save filtered files in the same directory

# Ensure target directory exists (it should, but good practice)
os.makedirs(target_dir, exist_ok=True)

# Files to process
splits = ["train", "val", "test"]
files_created = []
errors_occurred = False

for split in splits:
    source_file = os.path.join(source_dir, f"{split}.csv")
    target_file = os.path.join(target_dir, f"crema_d_{split}.csv")

    try:
        # Read the original CSV
        print(f"Reading {source_file}...")
        df = pd.read_csv(source_file)
        print(f"Read {len(df)} rows.")

        # Filter for CREMA-D dataset
        print(f"Filtering for dataset == 'crema_d'...")
        df_crema_d = df[df['dataset'] == 'crema_d'].copy() # Use .copy() to avoid SettingWithCopyWarning
        print(f"Found {len(df_crema_d)} CREMA-D rows.")

        # Save the filtered DataFrame
        if not df_crema_d.empty:
            print(f"Saving filtered data to: {target_file}...")
            df_crema_d.to_csv(target_file, index=False)
            print(f"Successfully saved {target_file}")
            files_created.append(target_file)
        else:
            print(f"No CREMA-D data found in {source_file}. Skipping {target_file}.")

    except FileNotFoundError:
        print(f"Error: Source file not found - {source_file}", file=sys.stderr)
        errors_occurred = True
    except Exception as e:
        print(f"An error occurred while processing {source_file}: {e}", file=sys.stderr)
        errors_occurred = True

print("\n--- Summary ---")
if files_created:
    print("Successfully created:")
    for f in files_created:
        print(f"- {f}")
else:
    print("No CREMA-D specific files were created.")

if errors_occurred:
    print("\nErrors occurred during processing. Please check the output above.", file=sys.stderr)
    sys.exit(1) # Exit with error code if issues occurred
else:
    print("\nFinished creating CREMA-D specific split files successfully.")
