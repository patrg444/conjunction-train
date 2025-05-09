import numpy as np
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Inspect shapes of embeddings in an NPZ file.")
    parser.add_argument("npz_path", help="Path to the .npz file")
    parser.add_argument("--base_dir", default=None, help="Optional base directory if npz_path is relative")
    args = parser.parse_args()

    file_path = args.npz_path
    if args.base_dir and not os.path.isabs(file_path):
        file_path = os.path.join(args.base_dir, file_path)

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}", file=sys.stderr)
        sys.exit(1)

    try:
        print(f"Inspecting file: {file_path}")
        print("-" * 30)
        # Load the file, allowing pickle for potential older formats if needed
        data = np.load(file_path, allow_pickle=True)

        # Check if it's an NpzFile (archive) or a single ndarray
        if isinstance(data, np.lib.npyio.NpzFile):
            print("Type: NPZ Archive")
            keys_found = {'text_embedding': False, 'audio_embedding': False, 'video_embedding': False}
            for key in data.files:
                try:
                    array = data[key]
                    shape = array.shape
                    print(f"Key: '{key}', Shape: {shape}")
                    if len(shape) > 0:
                        print(f"  -> Feature Dimension (last): {shape[-1]}")
                    if key in keys_found:
                        keys_found[key] = True
                except Exception as e:
                    print(f"Key: '{key}', Error reading shape: {e}")
            if not all(keys_found.values()):
                print("Warning: NPZ file did not contain all expected keys (text_embedding, audio_embedding, video_embedding)")
        elif isinstance(data, np.ndarray):
            print("Type: Single NPY Array")
            shape = data.shape
            print(f"Shape: {shape}")
            if len(shape) > 0:
                print(f"  -> Feature Dimension (last): {shape[-1]}")
        else:
            print(f"Error: Loaded object is of unexpected type: {type(data)}")

        print("-" * 30)

    except Exception as e:
        print(f"Error loading file {file_path}: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
