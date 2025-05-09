#!/bin/bash
# Script to verify WAV2VEC feature detection on EC2 instance

# Get EC2 IP
EC2_IP=$(cat aws_instance_ip.txt)
SSH_KEY="$HOME/Downloads/gpu-key.pem"  # Adjust if your key path is different

echo "=== WAV2VEC FEATURE DETECTION VERIFICATION ==="
echo "Target EC2: $EC2_IP"
echo

# SSH into the server and run diagnostic commands
echo "Running feature detection diagnostics..."
ssh -i $SSH_KEY ubuntu@$EC2_IP << 'EOF'
  cd ~/emotion_recognition
  
  echo "1. Checking WAV2VEC feature directories:"
  echo "   - /data/wav2vec_features:"
  ls -la /data/wav2vec_features 2>/dev/null || echo "   Directory not found or empty"
  
  echo "   - /data/wav2vec_crema_d:"
  ls -la /data/wav2vec_crema_d 2>/dev/null || echo "   Directory not found or empty"
  
  echo
  echo "2. Counting WAV2VEC feature files:"
  find /data -name "*.npz" -path "*/wav2vec*" 2>/dev/null | wc -l
  
  echo
  echo "3. Checking file permissions:"
  if [ -d "/data/wav2vec_features" ]; then
    stat -c "%a %U:%G %n" /data/wav2vec_features
    stat -c "%a %U:%G %n" /data/wav2vec_features/*.npz 2>/dev/null | head -n 3
  fi
  
  if [ -d "/data/wav2vec_crema_d" ]; then
    stat -c "%a %U:%G %n" /data/wav2vec_crema_d
    stat -c "%a %U:%G %n" /data/wav2vec_crema_d/*.npz 2>/dev/null | head -n 3
  fi
  
  echo
  echo "4. Testing file access as current user:"
  if [ -d "/data/wav2vec_features" ]; then
    echo "   - Can read directory:" $(/bin/ls -la /data/wav2vec_features &>/dev/null && echo "Yes" || echo "No")
    
    SAMPLE_FILE=$(find /data/wav2vec_features -name "*.npz" 2>/dev/null | head -n 1)
    if [ ! -z "$SAMPLE_FILE" ]; then
      echo "   - Can read sample file:" $(python3 -c "import numpy as np; np.load('$SAMPLE_FILE', allow_pickle=True)" &>/dev/null && echo "Yes" || echo "No")
    else
      echo "   - No sample files found to test"
    fi
  fi
  
  echo
  echo "5. Testing file path detection in script:"
  cat > /tmp/test_wav2vec_paths.py << 'PYTHON_EOF'
import os
import glob

data_dirs = ['/data/wav2vec_features', '/data/wav2vec_crema_d']

for data_dir in data_dirs:
    print(f"Checking directory: {data_dir}")
    if os.path.exists(data_dir):
        print(f"  Directory exists: Yes")
        npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
        print(f"  Files found: {len(npz_files)}")
        if npz_files:
            print(f"  Example file: {os.path.basename(npz_files[0])}")
    else:
        print(f"  Directory exists: No")
PYTHON_EOF

  python3 /tmp/test_wav2vec_paths.py
  
  echo
  echo "6. Checking train_attn_crnn_v2.py data loading:"
  SCRIPT_PATH="scripts/train_attn_crnn_v2.py"
  if [ -f "$SCRIPT_PATH" ]; then
    echo "   Script exists, extracting data loading code..."
    grep -A 5 "load_wav2vec_data" "$SCRIPT_PATH"
  else
    echo "   Script not found: $SCRIPT_PATH"
  fi
  
  echo
  echo "7. Testing data_dirs argument path resolution:"
  cat > /tmp/test_wav2vec_arg_paths.py << 'PYTHON_EOF'
import os
import sys
import glob

# Simulate CLI arguments
data_dirs = ['/data/wav2vec_features', '/data/wav2vec_crema_d']

print(f"Testing paths that would be used in script:")
for data_dir in data_dirs:
    print(f"\nChecking: {data_dir}")
    abspath = os.path.abspath(data_dir)
    print(f"  Absolute path: {abspath}")
    print(f"  Path exists: {os.path.exists(abspath)}")
    
    # Test glob pattern
    pattern = os.path.join(abspath, "*.npz")
    files = glob.glob(pattern)
    print(f"  Files matching '{pattern}': {len(files)}")
    
    # Show sample
    if files:
        print(f"  First file: {files[0]}")
PYTHON_EOF

  python3 /tmp/test_wav2vec_arg_paths.py
EOF

echo
echo "=== VERIFICATION COMPLETE ==="
echo "Based on these results, we can determine if the WAV2VEC features"
echo "are properly detected by the scripts on the EC2 instance."
