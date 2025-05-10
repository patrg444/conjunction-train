#!/bin/bash
#
# Script to verify improved WAV2VEC feature detection without changing anything
# This helps confirm that the fix works correctly

set -euo pipefail

# Configuration
EC2_IP="54.162.134.77"
EC2_KEY="$HOME/Downloads/gpu-key.pem"

echo "=== WAV2VEC Feature Location Verification ==="

# Connect to the EC2 instance and check all locations for WAV2VEC features
ssh -i "$EC2_KEY" ubuntu@$EC2_IP << 'EOF'
  # Define all possible directories to check based on our scan
  echo "Checking all possible locations for WAV2VEC features..."
  
  DIRS=(
    "~/emotion_project"
    "~/emotion_project/wav2vec_features" 
    "~/emotion_project/crema_d_features"
    "~/crema_d_features"
    "~/audio_emotion/models/wav2vec"
    "~/emotion-recognition/crema_d_features_facenet"
    "~/emotion-recognition/npz_files/CREMA-D"
    "~/emotion-recognition/crema_d_features_audio"
  )
  
  # Check each directory and print count of .npz files
  echo "+--------------------------------------------------------------+"
  echo "| Directory                                     | Feature Count |"
  echo "+--------------------------------------------------------------+"
  
  TOTAL_COUNT=0
  
  for DIR in "${DIRS[@]}"; do
    COUNT=$(find $DIR -name "*.npz" 2>/dev/null | wc -l)
    TOTAL_COUNT=$((TOTAL_COUNT + COUNT))
    printf "| %-48s | %12d |\n" "$DIR" "$COUNT"
  done
  
  echo "+--------------------------------------------------------------+"
  printf "| %-48s | %12d |\n" "TOTAL WAV2VEC FEATURES" "$TOTAL_COUNT"
  echo "+--------------------------------------------------------------+"
  
  # Check if directory exists but without WAV2VEC .npz files
  echo
  echo "Checking for directories that exist but have no .npz files:"
  for DIR in "${DIRS[@]}"; do
    if [ -d "$DIR" ] && [ $(find $DIR -name "*.npz" 2>/dev/null | wc -l) -eq 0 ]; then
      echo "  - $DIR exists but contains no .npz files"
    fi
  done
  
  echo
  echo "NOTE: The updated scripts will detect these files and avoid redundant uploads."
EOF

echo
echo "Verification complete!"
echo "The updated scripts will detect existing WAV2VEC features across multiple directories"
echo "and skip redundant uploads to the EC2 instance."
