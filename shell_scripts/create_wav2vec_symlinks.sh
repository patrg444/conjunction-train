#!/bin/bash
# Direct script to create symlinks for WAV2VEC features
# This will create the target directories and symlinks without checking

set -euo pipefail

EC2_IP=$(cat aws_instance_ip.txt)
SSH_KEY="$HOME/Downloads/gpu-key.pem"

echo "=== Creating WAV2VEC Feature Symlinks ==="
echo "Creating target directories and symlinks directly"

# Connect to EC2 and run the commands
ssh -i "$SSH_KEY" ubuntu@$EC2_IP << 'EOF'
  # Create target directories
  sudo mkdir -p /data/wav2vec_features
  sudo mkdir -p /data/wav2vec_crema_d
  sudo mkdir -p ~/emotion_project/wav2vec_features
  
  # Set permissions 
  sudo chown -R ubuntu:ubuntu /data/wav2vec_features
  sudo chown -R ubuntu:ubuntu /data/wav2vec_crema_d
  
  # Create symlinks from all legacy directories to both target directories
  if [ -d "/home/ubuntu/audio_emotion/models/wav2vec" ]; then
    echo "Creating symlinks from /home/ubuntu/audio_emotion/models/wav2vec"
    find /home/ubuntu/audio_emotion/models/wav2vec -name "*.npz" -type f | while read file; do
      ln -sf "$file" "/data/wav2vec_features/$(basename "$file")"
      ln -sf "$file" "/data/wav2vec_crema_d/$(basename "$file")"
    done
  fi
  
  if [ -d "/home/ubuntu/emotion-recognition/crema_d_features_audio" ]; then
    echo "Creating symlinks from /home/ubuntu/emotion-recognition/crema_d_features_audio"
    find /home/ubuntu/emotion-recognition/crema_d_features_audio -name "*.npz" -type f | while read file; do
      ln -sf "$file" "/data/wav2vec_features/$(basename "$file")"
      ln -sf "$file" "/data/wav2vec_crema_d/$(basename "$file")"
    done
  fi
  
  if [ -d "/home/ubuntu/emotion-recognition/npz_files/CREMA-D" ]; then
    echo "Creating symlinks from /home/ubuntu/emotion-recognition/npz_files/CREMA-D"
    find /home/ubuntu/emotion-recognition/npz_files/CREMA-D -name "*.npz" -type f | while read file; do
      ln -sf "$file" "/data/wav2vec_features/$(basename "$file")"
      ln -sf "$file" "/data/wav2vec_crema_d/$(basename "$file")"
    done
  fi
  
  # Count symlinks created
  WAV2VEC_LINKS=$(find /data/wav2vec_features -type l | wc -l)
  CREMA_D_LINKS=$(find /data/wav2vec_crema_d -type l | wc -l)
  
  echo "Created $WAV2VEC_LINKS symlinks in /data/wav2vec_features"
  echo "Created $CREMA_D_LINKS symlinks in /data/wav2vec_crema_d"
  
  # Verify that target directories have symlinks that actually resolve to files
  VALID_LINKS_WAV2VEC=$(find -L /data/wav2vec_features -type f | wc -l)
  VALID_LINKS_CREMA_D=$(find -L /data/wav2vec_crema_d -type f | wc -l)
  
  echo "Verified $VALID_LINKS_WAV2VEC valid feature files in /data/wav2vec_features"
  echo "Verified $VALID_LINKS_CREMA_D valid feature files in /data/wav2vec_crema_d"
  
  if [ "$VALID_LINKS_WAV2VEC" -gt 0 ] && [ "$VALID_LINKS_CREMA_D" -gt 0 ]; then
    echo "✓ Symlinks successfully created and verified"
  else
    echo "✗ Failed to create valid symlinks"
  fi
EOF

echo
echo "Feature symlink creation complete."
echo "You can now run: ./fixed_stream_attn_crnn_monitor.sh -c"
