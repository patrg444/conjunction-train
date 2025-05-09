# test_humor_dataloader.py
import sys
import os
import torch # Make sure torch is imported if needed by dataset/placeholders
# Add the parent directory (conjunction-train) to the Python path
# Assumes the script is run from /home/ubuntu/conjunction-train/scripts
# Adjust if conjunction-train is not the root or script is elsewhere
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
print(f"Project root added to path: {project_root}")
print(f"Current sys.path: {sys.path}")


# Check if the dataloaders module can be found
try:
    from dataloaders.humor_dataset import HumorDataset
    print("Successfully imported HumorDataset.")
except ImportError as e:
    print(f"Failed to import HumorDataset: {e}")
    print("Please ensure dataloaders/humor_dataset.py exists and the project structure is correct.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    sys.exit(1)

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MANIFEST_PATH = "/home/ubuntu/datasets/manifests/humor_manifest.csv"
# Dataset root should be the directory containing the 'smile', 'ava', etc. folders
# based on the relative paths in the manifest.
DATASET_ROOT = "/home/ubuntu/datasets"

print("\n--- Starting Dataloader Test ---")
print(f"Manifest Path: {MANIFEST_PATH}")
print(f"Dataset Root: {DATASET_ROOT}")

# Verify manifest exists before attempting to load
if not os.path.exists(MANIFEST_PATH):
    print(f"ERROR: Manifest file not found at {MANIFEST_PATH}")
    sys.exit(1)
else:
    print("Manifest file found.")

try:
    # Instantiate the dataset
    # Using default duration, sr, fps as defined in the class
    dataset = HumorDataset(manifest_path=MANIFEST_PATH, dataset_root=DATASET_ROOT)

    print(f"\nSuccessfully initialized HumorDataset. Size: {len(dataset)}")

    if len(dataset) > 0:
        print("\nAttempting to load sample 0...")
        try:
            # This will use the PLACEHOLDER load functions within humor_dataset.py
            sample_0 = dataset[0]
            print("Successfully retrieved sample 0 (using placeholder load functions).")
            print("Sample keys:", sample_0.keys())
            print(" Audio shape:", sample_0['audio'].shape)
            print(" Video shape:", sample_0['video'].shape)
            print(" Laugh Label:", sample_0['laugh_label'])
            print(" Has Video:", sample_0['has_video'])
            print(" Source:", sample_0['source'])
        except Exception as e:
            print(f"Error loading sample 0: {e}")
            logging.error("Error during sample loading:", exc_info=True)

    else:
        print("Dataset is empty, cannot load sample.")

except FileNotFoundError as e:
    print(f"\nError initializing dataset: {e}")
    logging.error("Dataset initialization failed:", exc_info=True)
except ValueError as e:
    print(f"\nError initializing dataset (likely manifest issue): {e}")
    logging.error("Dataset initialization failed:", exc_info=True)
except Exception as e:
    print(f"\nAn unexpected error occurred during initialization or loading: {e}")
    logging.error("Unexpected error during test:", exc_info=True)

print("\n--- Dataloader Test Finished ---")
