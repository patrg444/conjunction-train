#!/bin/bash

# Script to efficiently download CNN feature files from EC2 to local machine
# using rsync (handles restart, shows progress, and is much faster than individual scp)

echo "Starting fast download of CNN feature files..."

# Ensure target directories exist
mkdir -p data/ravdess_features_cnn_fixed 
mkdir -p data/crema_d_features_cnn_fixed

# Download RAVDESS CNN features with progress indicator
echo "Downloading RAVDESS CNN features..."
rsync -avz --progress -e "ssh -i $HOME/Downloads/gpu-key.pem" \
  ubuntu@54.162.134.77:/home/ubuntu/emotion-recognition/data/ravdess_features_cnn_fixed/ \
  data/ravdess_features_cnn_fixed/

# Download CREMA-D CNN features with progress indicator
echo "Downloading CREMA-D CNN features..."
rsync -avz --progress -e "ssh -i $HOME/Downloads/gpu-key.pem" \
  ubuntu@54.162.134.77:/home/ubuntu/emotion-recognition/data/crema_d_features_cnn_fixed/ \
  data/crema_d_features_cnn_fixed/

echo "Download complete. Both dataset feature files are now available locally."
echo "Files in RAVDESS: $(find data/ravdess_features_cnn_fixed -type f | wc -l)"
echo "Files in CREMA-D: $(find data/crema_d_features_cnn_fixed -type f | wc -l)"
