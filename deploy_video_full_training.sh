#!/bin/bash
# Deploy the video emotion training scripts to an EC2 instance
# This script:
# 1. Copies all the required scripts to the EC2 instance
# 2. Sets up the required directories
# 3. Makes the scripts executable

set -e

# Configuration
SSH_KEY="~/Downloads/gpu-key.pem"
EC2_USER="ubuntu"
EC2_HOST="54.162.134.77"
REMOTE_BASE_DIR="/home/ubuntu"
REMOTE_SCRIPTS_DIR="${REMOTE_BASE_DIR}/scripts"
REMOTE_DATA_DIR="${REMOTE_BASE_DIR}/datasets"

# Check for SSH key
if [ ! -f $(eval echo ${SSH_KEY}) ]; then
    echo "Error: SSH key not found at ${SSH_KEY}"
    echo "Please update the SSH_KEY variable in this script."
    exit 1
fi

# Function to run a command on the remote server
remote_exec() {
    ssh -i $(eval echo ${SSH_KEY}) ${EC2_USER}@${EC2_HOST} "$1"
}

# Create remote directories
echo "Setting up directories on the remote server..."
remote_exec "mkdir -p ${REMOTE_SCRIPTS_DIR} ${REMOTE_DATA_DIR}/video_manifest ${REMOTE_BASE_DIR}/emotion_full_video"

# Copy the scripts to the remote server
echo "Copying scripts to the remote server..."
scp -i $(eval echo ${SSH_KEY}) scripts/generate_video_manifest.py scripts/train_video_full.py ${EC2_USER}@${EC2_HOST}:${REMOTE_SCRIPTS_DIR}/
scp -i $(eval echo ${SSH_KEY}) launch_video_full_training.sh monitor_video_training.sh ${EC2_USER}@${EC2_HOST}:${REMOTE_BASE_DIR}/

# Make the scripts executable
echo "Making scripts executable..."
remote_exec "chmod +x ${REMOTE_BASE_DIR}/launch_video_full_training.sh ${REMOTE_BASE_DIR}/monitor_video_training.sh"

# Check for Python dependencies
echo "Checking Python dependencies..."
MISSING_DEPS=$(remote_exec "pip3 list | grep -E 'torch|torchvision|opencv-python|pandas|scikit-learn|matplotlib|seaborn|tqdm'" || echo "missing")

if [[ "$MISSING_DEPS" == "missing" ]]; then
    echo "Installing required Python packages..."
    remote_exec "pip3 install torch torchvision opencv-python pandas scikit-learn matplotlib seaborn tqdm"
fi

# Check for RAVDESS and CREMA-D datasets
echo "Checking for datasets..."
RAVDESS_COUNT=$(remote_exec "find ${REMOTE_DATA_DIR}/ravdess_videos -name '*.mp4' | wc -l" || echo "0")
CREMAD_COUNT=$(remote_exec "find ${REMOTE_DATA_DIR}/crema_d_videos -name '*.flv' | wc -l" || echo "0")

echo "Found ${RAVDESS_COUNT} RAVDESS videos and ${CREMAD_COUNT} CREMA-D videos."

if [[ "$RAVDESS_COUNT" -eq "0" ]] || [[ "$CREMAD_COUNT" -eq "0" ]]; then
    echo "Warning: One or both datasets appear to be missing or empty."
    echo "Make sure the datasets are available at:"
    echo "  - ${REMOTE_DATA_DIR}/ravdess_videos/"
    echo "  - ${REMOTE_DATA_DIR}/crema_d_videos/"
fi

echo "Deployment completed successfully!"
echo ""
echo "To start training:"
echo "  1. SSH into your EC2 instance:"
echo "     ssh -i ${SSH_KEY} ${EC2_USER}@${EC2_HOST}"
echo "  2. Run the training script:"
echo "     cd ${REMOTE_BASE_DIR} && ./launch_video_full_training.sh"
echo ""
echo "To monitor training:"
echo "  - Use the monitoring script:"
echo "    cd ${REMOTE_BASE_DIR} && ./monitor_video_training.sh"
echo ""
echo "The training will run in a tmux session for persistence."
echo "You can attach to it with: tmux attach -t video_training"
