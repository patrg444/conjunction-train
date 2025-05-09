import pickle
import os
from collections import Counter

PICKLE_PATH = "datasets/humor_datasets/ur_funny/v2/humor_label_sdk.pkl"

print(f"--- Inspecting Labels in {PICKLE_PATH} ---")

if not os.path.exists(PICKLE_PATH):
    print(f"Error: Pickle file not found at {PICKLE_PATH}")
else:
    try:
        with open(PICKLE_PATH, 'rb') as f:
            labels = pickle.load(f)

        if isinstance(labels, dict):
            # Assuming the pickle contains a dictionary where values are labels
            label_values = list(labels.values())
            if label_values:
                label_counts = Counter(label_values)
                print("\nLabel counts:")
                for label, count in label_counts.items():
                    print(f"  Label {label}: {count}")
            else:
                print("Pickle file loaded, but the dictionary is empty.")
        else:
            print(f"Pickle file loaded, but expected a dictionary, got {type(labels)}")

    except Exception as e:
        print(f"An error occurred while loading or inspecting the pickle file: {e}")

print("\n--- Inspection Complete ---")
