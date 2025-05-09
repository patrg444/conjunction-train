import os
import pickle
import sys

def inspect_pickle(pickle_path):
    """Loads a pickle file and prints information about its contents."""
    if not os.path.exists(pickle_path):
        print(f"Error: Pickle file not found at {pickle_path}")
        return

    print(f"Loading pickle file from {pickle_path}...")
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)

        print(f"Successfully loaded data. Type: {type(data)}")

        if isinstance(data, dict):
            print(f"Number of entries (keys): {len(data)}")
            print("Example keys:")
            # Print first 5 keys
            for i, key in enumerate(data.keys()):
                if i >= 5:
                    break
                print(f"- {key}")

            print("\nStructure of first 3 entries:")
            # Print structure of first 3 entries
            for i, (key, value) in enumerate(data.items()):
                if i >= 3:
                    break
                print(f"Key: {key}")
                print(f"Value type: {type(value)}")
                if isinstance(value, dict):
                    print(f"Value keys: {list(value.keys())}")
                    # If there's a 'text' key, print its type and a snippet
                    if 'text' in value and isinstance(value['text'], str):
                         print(f"  'text' type: {type(value['text'])}")
                         print(f"  'text' snippet: {value['text'][:100]}...") # Print first 100 chars
                    elif 'text' in value:
                         print(f"  'text' type: {type(value['text'])}")
                print("-" * 20)

        # Add more inspection logic here if needed for other data types

    except Exception as e:
        print(f"Error loading or inspecting pickle file: {e}")

if __name__ == "__main__":
    # Define the path to the pickle file relative to the script's execution directory
    # Assuming the script is run from /home/ubuntu/conjunction-train/scripts
    # and the pickle file is at /home/ubuntu/conjunction-train/datasets/humor_datasets/ur_funny/language_sdk.pkl
    pickle_file_path = "../datasets/humor_datasets/ur_funny/language_sdk.pkl"
    inspect_pickle(pickle_file_path)
