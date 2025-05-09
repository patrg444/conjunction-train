import pickle
import argparse
import os

def inspect_pickle(file_path):
    """
    Loads a pickle file and prints its top-level keys and a sample of its contents.
    """
    print(f"Attempting to load pickle file: {file_path}")
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Successfully loaded {file_path}")

        if isinstance(data, dict):
            print("\nPickle file is a dictionary. Keys:")
            for key in data.keys():
                print(f"- {key}")
            
            # Print a sample of the first key's value if it's a dict or list
            if data.keys():
                first_key = list(data.keys())[0]
                print(f"\nSample of data for key '{first_key}':")
                value = data[first_key]
                if isinstance(value, dict):
                    print("  Value is a dictionary. First 5 items (or fewer):")
                    for i, (k, v_item) in enumerate(value.items()):
                        if i < 5:
                            print(f"    '{k}': {v_item}")
                        else:
                            break
                elif isinstance(value, list):
                    print("  Value is a list. First 5 items (or fewer):")
                    for i, item in enumerate(value):
                        if i < 5:
                            print(f"    {item}")
                        else:
                            break
                else:
                    print(f"  Value type: {type(value)}. Value: {str(value)[:200]}...") # Print a snippet
        elif isinstance(data, list):
            print("\nPickle file is a list. Length:", len(data))
            print("First 5 items (or fewer):")
            for i, item in enumerate(data):
                if i < 5:
                    print(item)
                else:
                    break
        else:
            print("\nPickle file type:", type(data))
            print(f"Data (snippet): {str(data)[:500]}...")

    except Exception as e:
        print(f"Error inspecting pickle file {file_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect the contents of a pickle file.")
    parser.add_argument("pickle_path", type=str, help="Path to the .pkl file to inspect.")
    args = parser.parse_args()

    inspect_pickle(args.pickle_path)
