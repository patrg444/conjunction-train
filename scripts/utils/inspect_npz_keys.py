import numpy as np
import argparse
from pathlib import Path

def inspect_npz(npz_path):
    """Loads an NPZ file and prints its keys."""
    npz_file = Path(npz_path)
    if not npz_file.exists():
        print(f"ERROR: NPZ file not found: {npz_file}")
        return

    try:
        print(f"Inspecting keys in: {npz_file.name}")
        with np.load(npz_file) as data:
            keys = list(data.keys())
            print(f"  Keys found: {keys}")

            # Optionally print shapes or types if needed for debugging
            # for key in keys:
            #     try:
            #         print(f"    - {key}: shape={data[key].shape}, dtype={data[key].dtype}")
            #     except Exception as e:
            #         print(f"    - {key}: Error accessing details - {e}")

    except Exception as e:
        print(f"ERROR: Failed to load or read NPZ file {npz_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect keys in an NPZ file.")
    parser.add_argument("npz_file", type=str, help="Path to the NPZ file to inspect.")
    args = parser.parse_args()

    inspect_npz(args.npz_file)
