#!/bin/bash
# Generate video manifest file on EC2 for emotion recognition datasets
# This script copies the generate_video_manifest.py script to EC2 and runs it

# Configuration
KEY=~/Downloads/gpu-key.pem
EC2_HOST="ubuntu@54.162.134.77"
RAVDESS_DIR="/home/ubuntu/datasets/ravdess_videos"
CREMA_DIR="/home/ubuntu/datasets/crema_d_videos"
MANIFEST_PATH="/home/ubuntu/datasets/video_manifest.csv"

# Echo with timestamp
function log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Make sure generate_video_manifest.py is executable
chmod +x scripts/generate_video_manifest.py

# Create remote directories
log "Checking remote dataset directories..."
ssh -i $KEY $EC2_HOST "ls -la $RAVDESS_DIR || echo 'RAVDESS directory not found!'"
ssh -i $KEY $EC2_HOST "ls -la $CREMA_DIR || echo 'CREMA-D directory not found!'"

# Copy the script to EC2
log "Copying manifest generator script to EC2..."
scp -i $KEY scripts/generate_video_manifest.py $EC2_HOST:/home/ubuntu/scripts/

# Run the script on EC2
log "Generating manifest file on EC2..."
ssh -i $KEY $EC2_HOST "python /home/ubuntu/scripts/generate_video_manifest.py \
    --ravdess_dir $RAVDESS_DIR \
    --crema_dir $CREMA_DIR \
    --output $MANIFEST_PATH"

# Verify manifest was created
log "Verifying manifest file creation..."
ssh -i $KEY $EC2_HOST "ls -la $MANIFEST_PATH || echo 'Manifest file not created!'"
ssh -i $KEY $EC2_HOST "wc -l $MANIFEST_PATH"

# Print sample of the manifest
log "Sample of the manifest file:"
ssh -i $KEY $EC2_HOST "head -n 10 $MANIFEST_PATH"

log "Manifest generation complete!"
