#!/bin/bash
# Deploy and run the debug tools for the video-only Facenet emotion recognition model

# Variables
EC2_HOST="ubuntu@18.208.166.91"  # EC2 instance address
LOCAL_DEBUG_SCRIPTS="scripts/debug_batch_inspection.py scripts/check_class_distribution.py"
REMOTE_PATH="/home/ubuntu/emotion-recognition"

echo "=== Deploying Facenet Debug Tools ==="
echo "Connecting to $EC2_HOST..."

# Ensure remote directories exist
ssh $EC2_HOST "mkdir -p $REMOTE_PATH/scripts"

# Upload debug scripts
echo "Uploading debug scripts..."
scp $LOCAL_DEBUG_SCRIPTS $EC2_HOST:$REMOTE_PATH/scripts/

# Make scripts executable
echo "Setting executable permissions..."
ssh $EC2_HOST "chmod +x $REMOTE_PATH/scripts/debug_batch_inspection.py $REMOTE_PATH/scripts/check_class_distribution.py"

# Run the class distribution analysis first to check for data imbalance issues
echo "Running class distribution analysis..."
ssh $EC2_HOST "cd $REMOTE_PATH && python3 scripts/check_class_distribution.py" | tee class_distribution_output.txt

# Run batch inspection to check actual batches and labels
echo "Running batch inspection (sample of 100 files)..."
ssh $EC2_HOST "cd $REMOTE_PATH && python3 scripts/debug_batch_inspection.py --sample_limit 100" | tee batch_inspection_output.txt

# Download the generated charts for local viewing
echo "Downloading generated charts..."
scp $EC2_HOST:$REMOTE_PATH/emotion_*.png .

echo "Debug analysis complete. Check class_distribution_output.txt and batch_inspection_output.txt for detailed results."
echo "Emotion distribution charts saved locally."
