#!/bin/bash
# Script to verify the WAV2VEC data structure on the EC2 server

# Check if the AWS instance IP is available
if [ ! -f "aws_instance_ip.txt" ]; then
  echo "Error: EC2 host IP not found. Please create aws_instance_ip.txt file."
  exit 1
fi

EC2_USER="ubuntu"
EC2_HOST=$(cat aws_instance_ip.txt)

echo "===== Verifying WAV2VEC Data Structure on EC2 ====="

# Create a short Python script to inspect the NPZ files
cat > inspect_npz.py << 'EOF'
#!/usr/bin/env python3
"""
Inspect the structure of WAV2VEC NPZ files to diagnose issues.
"""
import os
import sys
import glob
import numpy as np

# Search for WAV2VEC features in multiple possible locations
search_dirs = [
    "/home/ubuntu/audio_emotion/wav2vec_features",
    "/home/ubuntu/wav2vec_features",
    "/home/ubuntu/audio_emotion/features/wav2vec",
    "/home/ubuntu/features/wav2vec",
    "/data/wav2vec_features",
    "/home/ubuntu/wav2vec_sample/home/ubuntu/audio_emotion/models/wav2vec"
]

# Find all feature files
feature_files = []
for dir_path in search_dirs:
    if os.path.exists(dir_path):
        print(f"\nSearching in directory: {dir_path}")
        npz_files = glob.glob(os.path.join(dir_path, "*.npz"))
        if npz_files:
            feature_files.extend(npz_files)
            print(f"Found {len(npz_files)} feature files in {dir_path}")

# If no files found in common directories, try a wider search
if not feature_files:
    print("\nNo feature files found in specified directories. Trying a wider search...")
    for root_dir in ["/home/ubuntu", "/data"]:
        if os.path.exists(root_dir):
            print(f"Searching in {root_dir}...")
            try:
                # Using find command for efficiency in large directories
                import subprocess
                cmd = f"find {root_dir} -name '*.npz' | head -n 100"
                result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
                found_files = result.stdout.strip().split('\n')
                if found_files and found_files[0]:
                    feature_files.extend(found_files)
                    print(f"Found {len(found_files)} feature files in {root_dir}")
            except Exception as e:
                print(f"Error searching {root_dir}: {str(e)}")

# Inspect the structure of the first few files
print("\n===== NPZ File Structure Analysis =====")
if not feature_files:
    print("No feature files found.")
    sys.exit(1)

# Analyze a sample of files
sample_size = min(10, len(feature_files))
print(f"\nAnalyzing structure of {sample_size} sample files:")

for i, file_path in enumerate(feature_files[:sample_size]):
    print(f"\nFile {i+1}: {file_path}")
    try:
        data = np.load(file_path, allow_pickle=True)
        print(f"  Keys: {list(data.keys())}")
        
        # Check file contents for each key
        for key in data.keys():
            try:
                value = data[key]
                if isinstance(value, np.ndarray):
                    print(f"  {key}: numpy array, shape {value.shape}, dtype {value.dtype}")
                else:
                    print(f"  {key}: {type(value)}, value: {value}")
            except Exception as e:
                print(f"  Error accessing key {key}: {str(e)}")
    except Exception as e:
        print(f"  Error loading file: {str(e)}")

# Count files with expected structure
print("\n===== Structure Statistics =====")
has_embeddings = 0
has_label = 0
other_keys = set()

for file_path in feature_files:
    try:
        data = np.load(file_path, allow_pickle=True)
        if 'embeddings' in data:
            has_embeddings += 1
        if 'label' in data:
            has_label += 1
        for key in data.keys():
            if key not in ['embeddings', 'label']:
                other_keys.add(key)
    except:
        continue

print(f"Total files analyzed: {len(feature_files)}")
print(f"Files with 'embeddings' key: {has_embeddings} ({has_embeddings/len(feature_files)*100:.1f}%)")
print(f"Files with 'label' key: {has_label} ({has_label/len(feature_files)*100:.1f}%)")
if other_keys:
    print(f"Other keys found: {', '.join(other_keys)}")

# Suggest next steps
print("\n===== Recommendations =====")
if has_embeddings == 0:
    print("ISSUE: No files contain the expected 'embeddings' key.")
    print("Possible fixes:")
    print("1. Check if the files use a different key name for the embeddings.")
    print("2. Verify that WAV2VEC feature extraction was completed properly.")
    print("3. Look for WAV2VEC features in other directories or servers.")
    print("4. Re-run the WAV2VEC feature extraction process.")
else:
    print(f"Found {has_embeddings} files with the expected 'embeddings' key.")
    if has_embeddings < len(feature_files):
        print(f"Warning: {len(feature_files) - has_embeddings} files are missing the 'embeddings' key.")
EOF

# Upload the script to the server
scp -i ~/Downloads/gpu-key.pem -o StrictHostKeyChecking=no inspect_npz.py $EC2_USER@$EC2_HOST:~/audio_emotion/inspect_npz.py

# Make it executable
ssh -i ~/Downloads/gpu-key.pem -o StrictHostKeyChecking=no $EC2_USER@$EC2_HOST "chmod +x ~/audio_emotion/inspect_npz.py"

# Run the inspection script
echo "Running inspection script on EC2..."
ssh -i ~/Downloads/gpu-key.pem -o StrictHostKeyChecking=no $EC2_USER@$EC2_HOST "cd ~/audio_emotion && python3 inspect_npz.py > npz_inspection_results.txt"

# Download the results
echo "Downloading results..."
mkdir -p data_verification
scp -i ~/Downloads/gpu-key.pem -o StrictHostKeyChecking=no $EC2_USER@$EC2_HOST:~/audio_emotion/npz_inspection_results.txt data_verification/

# Check if results were downloaded successfully
if [ -f "data_verification/npz_inspection_results.txt" ]; then
  echo "Inspection results downloaded successfully."
  echo "Results saved to data_verification/npz_inspection_results.txt"
  
  # Display summary
  echo -e "\n===== Data Inspection Summary ====="
  grep -A 10 "Structure Statistics" data_verification/npz_inspection_results.txt
  echo -e "\n===== Recommendations ====="
  grep -A 10 "Recommendations" data_verification/npz_inspection_results.txt
else
  echo "Warning: Could not download inspection results."
fi

echo "===== WAV2VEC Data Verification Complete ====="
