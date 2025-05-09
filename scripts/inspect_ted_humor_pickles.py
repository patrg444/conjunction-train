import pickle
import os
import numpy as np

dataset_path = "datasets/humor_datasets/ted_humor_sdk_v1/final_humor_sdk"

files_to_inspect = [
    "data_folds.pkl",
    "humor_label_sdk.pkl",
    "covarep_features_sdk.pkl",
    "openface_features_sdk.pkl",
    "word_embedding_sdk.pkl", # Assuming this exists based on word_embedding_list/indexes
    "word_embedding_list.pkl",
    "word_embedding_indexes_sdk.pkl"
]

print(f"Inspecting pickle files in {dataset_path}")

for filename in files_to_inspect:
    filepath = os.path.join(dataset_path, filename)
    print(f"\n--- Inspecting {filename} ---")
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        continue

    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        print(f"Type: {type(data)}")

        if isinstance(data, dict):
            print(f"Keys: {data.keys()}")
            # Optionally print info about the first few items in dict values
            for key, value in list(data.items())[:3]: # Look at first 3 keys
                print(f"  Key '{key}': Type: {type(value)}")
                if isinstance(value, np.ndarray):
                    print(f"    Shape: {value.shape}")
                    print(f"    Dtype: {value.dtype}")
                elif isinstance(value, list):
                    print(f"    Length: {len(value)}")
                    if len(value) > 0:
                        print(f"    Type of first element: {type(value[0])}")
                        if isinstance(value[0], np.ndarray):
                             print(f"    Shape of first element: {value[0].shape}")
                             print(f"    Dtype of first element: {value[0].dtype}")
                elif isinstance(value, dict):
                     print(f"    Keys of first element (if dict): {list(value.keys())[:5]}")


        elif isinstance(data, list):
            print(f"Length: {len(data)}")
            if len(data) > 0:
                print(f"Type of first element: {type(data[0])}")
                if isinstance(data[0], np.ndarray):
                    print(f"Shape of first element: {data[0].shape}")
                    print(f"Dtype of first element: {data[0].dtype}")
                elif isinstance(data[0], dict):
                     print(f"Keys of first element (if dict): {list(data[0].keys())[:5]}")

        elif isinstance(data, np.ndarray):
            print(f"Shape: {data.shape}")
            print(f"Dtype: {data.dtype}")

    except Exception as e:
        print(f"Error loading or inspecting {filename}: {e}")

print("\n--- Inspection Complete ---")
