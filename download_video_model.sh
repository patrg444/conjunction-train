#!/bin/bash
# Download the trained video emotion model and results from the EC2 instance
# This script:
# 1. Finds the best model and evaluation results
# 2. Downloads them to the local machine

set -e

# Configuration
SSH_KEY="~/Downloads/gpu-key.pem"
EC2_USER="ubuntu"
EC2_HOST="54.162.134.77"
REMOTE_DIR="/home/ubuntu/emotion_full_video"
LOCAL_DIR="./downloaded_models"

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

# Check if the training has been run
if ! remote_exec "test -d ${REMOTE_DIR} && ls ${REMOTE_DIR}/*/model_best.pt >/dev/null 2>&1"; then
    echo "Error: No trained models found on the EC2 instance."
    echo "Please make sure training has been completed first."
    exit 1
fi

# Create local directory
mkdir -p ${LOCAL_DIR}

# Find the best model directory
echo "Finding best model on EC2 instance..."
BEST_MODEL_DIR=$(remote_exec "ls -td ${REMOTE_DIR}/video_full_* | head -1")
echo "Found best model at: ${BEST_MODEL_DIR}"

# Get the model timestamp
MODEL_TIMESTAMP=$(basename ${BEST_MODEL_DIR})
LOCAL_MODEL_DIR="${LOCAL_DIR}/${MODEL_TIMESTAMP}"
mkdir -p ${LOCAL_MODEL_DIR}

# Download the model and evaluation results
echo "Downloading model and results..."
rsync -avP -e "ssh -i $(eval echo ${SSH_KEY})" \
    ${EC2_USER}@${EC2_HOST}:${BEST_MODEL_DIR}/model_best.pt \
    ${LOCAL_MODEL_DIR}/

rsync -avP -e "ssh -i $(eval echo ${SSH_KEY})" \
    ${EC2_USER}@${EC2_HOST}:${BEST_MODEL_DIR}/metrics.json \
    ${LOCAL_MODEL_DIR}/

rsync -avP -e "ssh -i $(eval echo ${SSH_KEY})" \
    ${EC2_USER}@${EC2_HOST}:${BEST_MODEL_DIR}/learning_curves.png \
    ${LOCAL_MODEL_DIR}/

rsync -avP -e "ssh -i $(eval echo ${SSH_KEY})" \
    ${EC2_USER}@${EC2_HOST}:${BEST_MODEL_DIR}/args.json \
    ${LOCAL_MODEL_DIR}/

# Download the last confusion matrix
rsync -avP -e "ssh -i $(eval echo ${SSH_KEY})" \
    ${EC2_USER}@${EC2_HOST}:"${BEST_MODEL_DIR}/confusion_matrix_*.png" \
    ${LOCAL_MODEL_DIR}/ 2>/dev/null || echo "No confusion matrices found."

# Download the last classification report
rsync -avP -e "ssh -i $(eval echo ${SSH_KEY})" \
    ${EC2_USER}@${EC2_HOST}:"${BEST_MODEL_DIR}/classification_report_*.json" \
    ${LOCAL_MODEL_DIR}/ 2>/dev/null || echo "No classification reports found."

# Create a summary file
echo "Creating summary..."
echo "# Video Emotion Model Summary" > ${LOCAL_MODEL_DIR}/summary.md
echo "Downloaded on: $(date)" >> ${LOCAL_MODEL_DIR}/summary.md
echo "" >> ${LOCAL_MODEL_DIR}/summary.md

# Extract best validation accuracy
if [ -f ${LOCAL_MODEL_DIR}/metrics.json ]; then
    BEST_ACC=$(grep -o '"best_val_acc": [0-9.]*' ${LOCAL_MODEL_DIR}/metrics.json | cut -d' ' -f2)
    echo "Best validation accuracy: ${BEST_ACC}%" >> ${LOCAL_MODEL_DIR}/summary.md
fi

# Add model parameters
if [ -f ${LOCAL_MODEL_DIR}/args.json ]; then
    echo "" >> ${LOCAL_MODEL_DIR}/summary.md
    echo "## Model Parameters" >> ${LOCAL_MODEL_DIR}/summary.md
    echo '```json' >> ${LOCAL_MODEL_DIR}/summary.md
    cat ${LOCAL_MODEL_DIR}/args.json >> ${LOCAL_MODEL_DIR}/summary.md
    echo '```' >> ${LOCAL_MODEL_DIR}/summary.md
fi

echo ""
echo "Download completed successfully!"
echo "Model and results saved to: ${LOCAL_MODEL_DIR}"
