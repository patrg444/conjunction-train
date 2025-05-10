#!/bin/bash
# Check the keys in the NPZ files

# Define variables
EC2_HOST="ubuntu@54.162.134.77"
KEY_PATH="~/Downloads/gpu-key.pem"
REMOTE_DIR="/home/ubuntu/audio_emotion/models/wav2vec"

echo "Examining NPZ file structure..."
echo "==============================="

# Create a temporary Python script to check the keys in NPZ files
ssh -i $KEY_PATH $EC2_HOST "cat > /tmp/check_npz_keys.py << 'EOL'
#!/usr/bin/env python3
import os
import sys
import numpy as np

# Directory with NPZ files
wav2vec_dir = '/home/ubuntu/audio_emotion/models/wav2vec'

# Get first files from each dataset
cremad_file = None
ravdess_file = None

for filename in os.listdir(wav2vec_dir):
    if filename.endswith('.npz'):
        if filename.startswith('cremad_') and cremad_file is None:
            cremad_file = os.path.join(wav2vec_dir, filename)
        elif filename.startswith('ravdess_') and ravdess_file is None:
            ravdess_file = os.path.join(wav2vec_dir, filename)
    
    if cremad_file and ravdess_file:
        break

# Check keys in CREMA-D file
if cremad_file:
    print(f'\\nCREMA-D file: {os.path.basename(cremad_file)}')
    data = np.load(cremad_file)
    print(f'Keys in file: {list(data.keys())}')
    for key in data.keys():
        print(f'  {key}: shape={data[key].shape}, dtype={data[key].dtype}')

# Check keys in RAVDESS file
if ravdess_file:
    print(f'\\nRAVDESS file: {os.path.basename(ravdess_file)}')
    data = np.load(ravdess_file)
    print(f'Keys in file: {list(data.keys())}')
    for key in data.keys():
        print(f'  {key}: shape={data[key].shape}, dtype={data[key].dtype}')
EOL"

# Execute the Python script
echo "Running key check script on server..."
ssh -i $KEY_PATH $EC2_HOST "chmod +x /tmp/check_npz_keys.py && python3 /tmp/check_npz_keys.py"
