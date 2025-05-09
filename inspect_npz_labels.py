import numpy as np
import sys
from collections import Counter

def inspect_labels(npz_path):
    try:
        data = np.load(npz_path)
        if 'labels' not in data:
            print(f"ERROR: 'labels' key not found in {npz_path}")
            return

        labels = data['labels']
        print(f"Inspecting labels in: {npz_path}")
        print(f"Total labels found: {len(labels)}")

        unique_labels, counts = np.unique(labels, return_counts=True)
        label_counts = dict(zip(unique_labels, counts))

        print("Unique labels and their counts:")
        # Sort by label index for clarity
        for label in sorted(label_counts.keys()):
            print(f"  Label {label}: {label_counts[label]} times")

    except FileNotFoundError:
        print(f"ERROR: File not found at {npz_path}")
    except Exception as e:
        print(f"ERROR: Failed to load or process {npz_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_npz_labels.py <path_to_npz_file>")
        sys.exit(1)
    
    npz_file_path = sys.argv[1]
    inspect_labels(npz_file_path)
