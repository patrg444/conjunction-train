#!/bin/bash
# Script to create symlinks only from the primary WAV2VEC features directory
# This avoids duplicates and uses only the correct 8690 files

set -euo pipefail

EC2_IP=$(cat aws_instance_ip.txt)
SSH_KEY="$HOME/Downloads/gpu-key.pem"

echo "=== Creating WAV2VEC Feature Symlinks (Primary Source Only) ==="
echo "This script will only create symlinks from the primary source directory"
echo "with the correct 8690 files the model needs."

# Connect to EC2 and run the commands
ssh -i "$SSH_KEY" ubuntu@$EC2_IP << 'EOF'
  # First clean any existing symlinks to avoid duplication
  echo "Cleaning existing target directories..."
  sudo mkdir -p /data/wav2vec_features
  sudo mkdir -p /data/wav2vec_crema_d
  sudo chown -R ubuntu:ubuntu /data/wav2vec_features
  sudo chown -R ubuntu:ubuntu /data/wav2vec_crema_d
  
  # Remove any existing symlinks
  find /data/wav2vec_features -type l -delete
  find /data/wav2vec_crema_d -type l -delete
  
  # Only create symlinks from the primary WAV2VEC features directory
  PRIMARY_SOURCE="/home/ubuntu/audio_emotion/models/wav2vec"
  
  if [ -d "$PRIMARY_SOURCE" ]; then
    echo "Creating symlinks from $PRIMARY_SOURCE only..."
    
    # Count files first
    FILE_COUNT=$(find "$PRIMARY_SOURCE" -name "*.npz" -type f | wc -l)
    echo "Found $FILE_COUNT feature files in primary source directory"
    
    # Create symlinks to both target directories
    echo "Creating symlinks to target directories..."
    find "$PRIMARY_SOURCE" -name "*.npz" -type f | while read file; do
      ln -sf "$file" "/data/wav2vec_features/$(basename "$file")"
      ln -sf "$file" "/data/wav2vec_crema_d/$(basename "$file")"
    done
  else
    echo "ERROR: Primary source directory not found: $PRIMARY_SOURCE"
    exit 1
  fi
  
  # Verify created symlinks
  WAV2VEC_LINKS=$(find /data/wav2vec_features -type l | wc -l)
  CREMA_D_LINKS=$(find /data/wav2vec_crema_d -type l | wc -l)
  
  echo "Created $WAV2VEC_LINKS symlinks in /data/wav2vec_features"
  echo "Created $CREMA_D_LINKS symlinks in /data/wav2vec_crema_d"
  
  # Verify that target directories have symlinks that actually resolve to files
  VALID_LINKS_WAV2VEC=$(find -L /data/wav2vec_features -type f | wc -l)
  VALID_LINKS_CREMA_D=$(find -L /data/wav2vec_crema_d -type f | wc -l)
  
  echo "Verified $VALID_LINKS_WAV2VEC valid feature files in /data/wav2vec_features"
  echo "Verified $VALID_LINKS_CREMA_D valid feature files in /data/wav2vec_crema_d"
  
  if [ "$VALID_LINKS_WAV2VEC" -eq 8690 ] && [ "$VALID_LINKS_CREMA_D" -eq 8690 ]; then
    echo "✓ Exactly 8690 symlinks successfully created and verified in each target directory"
  else
    echo "✗ Warning: Did not create exactly 8690 symlinks. Please investigate."
  fi
EOF

echo
echo "Feature symlink creation complete."
echo "You can now run: ./fixed_stream_attn_crnn_monitor.sh -c"
