#!/bin/bash
#
# Simple script to verify WAV2VEC features on EC2
#

set -eo pipefail

# Configuration
IP="54.162.134.77"
PEM="$HOME/Downloads/gpu-key.pem"

echo "=== VERIFYING WAV2VEC FEATURES ==="

# Connect to EC2 to check features
ssh -i "$PEM" ubuntu@$IP << 'EOF'
  # Check symlink first
  echo "Checking symlink..."
  if [ -L "/home/ubuntu/emotion_project/wav2vec_features" ]; then
    echo "Symlink exists at /home/ubuntu/emotion_project/wav2vec_features"
    ls -la /home/ubuntu/emotion_project/wav2vec_features | head -n 5
  else
    echo "Symlink does not exist"
  fi

  # Check WAV2VEC features in primary location
  echo -e "\nChecking primary WAV2VEC location..."
  COUNT=$(find -L /home/ubuntu/audio_emotion/models/wav2vec -name "*.npz" 2>/dev/null | wc -l)
  echo "Found $COUNT WAV2VEC feature files in audio_emotion/models/wav2vec"
  ls -la /home/ubuntu/audio_emotion/models/wav2vec | head -n 5
  
  # Check through symlink
  echo -e "\nChecking through symlink..."
  COUNT=$(find -L /home/ubuntu/emotion_project/wav2vec_features -name "*.npz" 2>/dev/null | wc -l)
  echo "Found $COUNT WAV2VEC feature files through the symlink"
  ls -la /home/ubuntu/emotion_project/wav2vec_features | head -n 5
EOF

echo "=== VERIFICATION COMPLETE ==="
