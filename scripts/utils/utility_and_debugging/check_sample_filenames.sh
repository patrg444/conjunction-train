#!/bin/bash
# Just show sample filenames to understand the real pattern

# Define variables
EC2_HOST="ubuntu@54.162.134.77"
KEY_PATH="~/Downloads/gpu-key.pem"
REMOTE_DIR="/home/ubuntu/audio_emotion/models/wav2vec"

echo "Checking sample filenames in wav2vec directory..."
echo "================================================="

# Show first 20 filenames to understand pattern
echo "First 20 filenames:"
ssh -i $KEY_PATH $EC2_HOST "ls -1 $REMOTE_DIR/*.npz | head -20"

# Check if there might be a metadata file that describes the dataset
echo -e "\nChecking for metadata or README files:"
ssh -i $KEY_PATH $EC2_HOST "find $REMOTE_DIR -type f -name \"*.txt\" -o -name \"*.csv\" -o -name \"*.json\" -o -name \"README*\""
