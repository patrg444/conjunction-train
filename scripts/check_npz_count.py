# check_npz_count.py
import numpy as np
import sys

npz_path = 'corrected_manifest_hubert_embeddings.npz'

try:
    data = np.load(npz_path)
    if 'embeddings' in data:
        count = data['embeddings'].shape[0]
        print(f"Embeddings count in {npz_path}: {count}")
    else:
        print(f"Error: 'embeddings' key not found in {npz_path}")
        sys.exit(1)
except Exception as e:
    print(f"Error loading or checking {npz_path}: {e}")
    sys.exit(1)
