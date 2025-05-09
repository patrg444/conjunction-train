#!/bin/bash
#
# Improved script to prepare WAV2VEC data for ATTN-CRNN training
# Properly checks if features already exist on EC2 before uploading

set -euo pipefail

# Get EC2 info from aws_instance_ip.txt
EC2_IP=$(cat aws_instance_ip.txt)
SSH_KEY="$HOME/Downloads/gpu-key.pem"  # Adjust if your key path is different
LOCAL_DATASET="${1:-crema_d}"  # Default to crema_d if not specified

# Target directories on EC2 that our ATTN-CRNN v2 model expects
TARGET_DIRS=(
  "/data/wav2vec_features"
  "/data/wav2vec_crema_d"
)

# Additional directories where features might exist but need symlinking
LEGACY_DIRS=(
  "/home/ubuntu/emotion_project/wav2vec_features"
  "/home/ubuntu/audio_emotion/models/wav2vec"
  "/home/ubuntu/emotion-recognition/crema_d_features_audio"
  "/home/ubuntu/emotion-recognition/crema_d_features"
  "/home/ubuntu/emotion-recognition/npz_files/CREMA-D"
)

echo "=== WAV2VEC Feature Setup ==="
echo "Setting up data for $LOCAL_DATASET dataset"
echo "Target EC2 instance: $EC2_IP"

# Function to thoroughly check if WAV2VEC features already exist on EC2
check_remote_features() {
  echo "Performing thorough check for existing WAV2VEC features on EC2..."
  
  # First check target directories
  local found_in_target=false
  local total_count=0
  local target_count=0
  
  echo "Checking target directories expected by ATTN-CRNN v2:"
  for dir in "${TARGET_DIRS[@]}"; do
    echo "  - Checking $dir..."
    local count=$(ssh -i "$SSH_KEY" ubuntu@$EC2_IP "find $dir -name '*.npz' 2>/dev/null | wc -l" || echo "0")
    
    if [ "$count" -gt 0 ]; then
      echo "    ✓ Found $count feature files in $dir"
      found_in_target=true
      target_count=$((target_count + count))
    else
      echo "    ✗ No feature files found in $dir"
    fi
    
    total_count=$((total_count + count))
  done
  
  # Then check legacy directories
  echo "Checking legacy directories where features might exist:"
  local found_in_legacy=false
  local legacy_count=0
  
  for dir in "${LEGACY_DIRS[@]}"; do
    echo "  - Checking $dir..."
    local count=$(ssh -i "$SSH_KEY" ubuntu@$EC2_IP "find $dir -name '*.npz' 2>/dev/null | wc -l" || echo "0")
    
    if [ "$count" -gt 0 ]; then
      echo "    ✓ Found $count feature files in $dir"
      found_in_legacy=true
      legacy_count=$((legacy_count + count))
    else
      echo "    ✗ No feature files found in $dir"
    fi
    
    total_count=$((total_count + count))
  done
  
  echo
  if [ "$total_count" -gt 0 ]; then
    echo "✓ Total WAV2VEC feature files found on EC2: $total_count"
    
    if [ "$found_in_target" = true ]; then
      echo "  ✓ $target_count files already in target directories"
      
      if [ "$found_in_legacy" = true ]; then
        echo "  ✓ $legacy_count files in legacy directories (will be symlinked)"
        return 3  # Found in both target and legacy
      else
        return 1  # Found only in target
      fi
      
    elif [ "$found_in_legacy" = true ]; then
      echo "  ✓ $legacy_count files in legacy directories (will be symlinked)"
      return 2  # Found only in legacy
    fi
  else
    echo "✗ No WAV2VEC feature files found on EC2 in any directory"
    return 0  # Not found anywhere
  fi
}

# Function to create symlinks from legacy to target directories
create_symlinks() {
  echo "Creating symlinks from legacy to target directories..."
  
  # Create target directories if they don't exist
  for dir in "${TARGET_DIRS[@]}"; do
    ssh -i "$SSH_KEY" ubuntu@$EC2_IP "sudo mkdir -p $dir && sudo chmod 777 $dir"
  done
  
  # For each legacy directory, create symlinks to target directories
  for legacy_dir in "${LEGACY_DIRS[@]}"; do
    # Get count to check if directory has features
    local count=$(ssh -i "$SSH_KEY" ubuntu@$EC2_IP "find $legacy_dir -name '*.npz' 2>/dev/null | wc -l" || echo "0")
    
    if [ "$count" -gt 0 ]; then
      echo "  - Linking $count files from $legacy_dir..."
      
      # Determine target directory based on directory name
      local target_dir="${TARGET_DIRS[0]}"  # Default to first target
      if [[ "$legacy_dir" == *"crema_d"* ]]; then
        target_dir="${TARGET_DIRS[1]}"  # Use CREMA-D specific target
      fi
      
      # Create symlinks (single file at a time to avoid wildcard expansion issues)
      ssh -i "$SSH_KEY" ubuntu@$EC2_IP "find $legacy_dir -name '*.npz' -type f | while read file; do ln -sf \"\$file\" $target_dir/\$(basename \"\$file\"); done"
      
      echo "    ✓ Created symlinks in $target_dir"
    fi
  done
  
  # Verify symlinks were created
  for dir in "${TARGET_DIRS[@]}"; do
    local count=$(ssh -i "$SSH_KEY" ubuntu@$EC2_IP "find $dir -type l | wc -l" || echo "0")
    echo "  ✓ Created $count symlinks in $dir"
  done
  
  echo "✓ Symlink creation complete"
}

# Function to extract and upload WAV2VEC features
upload_features() {
  echo "Preparing WAV2VEC features for upload..."
  
  # Check for local crema_d_features directory first
  if [ -d "crema_d_features" ] && [ "$(find crema_d_features -name '*.npz' 2>/dev/null | wc -l)" -gt 0 ]; then
    local count=$(find crema_d_features -name '*.npz' 2>/dev/null | wc -l)
    echo "✓ Found $count local WAV2VEC features in crema_d_features directory"
    
    # Create a tarball of existing features
    echo "Creating tarball for upload..."
    tar -czf wav2vec_features.tar.gz crema_d_features
    
    # Ensure target directories exist
    ssh -i "$SSH_KEY" ubuntu@$EC2_IP "sudo mkdir -p ${TARGET_DIRS[1]} && sudo chmod 777 ${TARGET_DIRS[1]}"
    
    # Upload to the EC2 instance
    echo "Uploading WAV2VEC features to EC2 (this may take a while)..."
    scp -i "$SSH_KEY" wav2vec_features.tar.gz ubuntu@$EC2_IP:/tmp/
    
    # Extract on the remote machine directly to the target directory
    echo "Extracting features on EC2..."
    ssh -i "$SSH_KEY" ubuntu@$EC2_IP "cd ${TARGET_DIRS[1]} && tar -xzf /tmp/wav2vec_features.tar.gz --strip-components=1 && rm /tmp/wav2vec_features.tar.gz"
    
    # Verify upload was successful
    local remote_count=$(ssh -i "$SSH_KEY" ubuntu@$EC2_IP "find ${TARGET_DIRS[1]} -name '*.npz' | wc -l")
    echo "✓ Verified $remote_count WAV2VEC feature files now on EC2"
    
    return 0
  else
    echo "✗ No local WAV2VEC features found in crema_d_features directory"
    return 1
  fi
}

# Main execution starts here
check_remote_features
FEATURE_STATUS=$?

case $FEATURE_STATUS in
  0)
    echo
    echo "No WAV2VEC features found on EC2. Need to upload features."
    echo
    if upload_features; then
      echo "✓ Successfully uploaded WAV2VEC features to EC2"
    else
      echo "✗ Failed to upload WAV2VEC features"
      echo
      echo "Warning: No WAV2VEC features available. Training may fail."
      echo "Please ensure features are available in crema_d_features directory"
      echo "or on the EC2 instance before running training."
    fi
    ;;
  1)
    echo
    echo "✓ WAV2VEC features already exist in target directories on EC2."
    echo "  No need to upload features. Ready to train."
    ;;
  2)
    echo
    echo "WAV2VEC features exist in legacy directories but not in target directories."
    echo "Creating symlinks to make them accessible in target directories..."
    echo
    create_symlinks
    echo
    echo "✓ Features are now accessible in target directories via symlinks."
    echo "  Ready to train."
    ;;
  3)
    echo
    echo "WAV2VEC features exist in both target and legacy directories."
    echo "Creating additional symlinks from legacy directories..."
    echo
    create_symlinks
    echo
    echo "✓ All features are now accessible. Ready to train."
    ;;
esac

echo
echo "=== SETUP COMPLETE ==="
echo "You can now run the ATTN-CRNN v2 training with:"
echo "  ./deploy_attn_crnn_v2.sh"
echo
echo "To monitor training progress:"
echo "  ./monitor_attn_crnn_v2.sh -c"
