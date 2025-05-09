#!/bin/bash
# Script to download the trained combined model from AWS

INSTANCE_IP="3.235.76.0"
USERNAME="ec2-user"
KEY_FILE="./aws-setup/emotion-recognition-key-fixed-20250323090016.pem"
REMOTE_MODEL_DIR="~/emotion_training/models/branched_regularization_sync_aug"
LOCAL_MODEL_DIR="./models/branched_regularization_sync_aug"

echo "=================================================================="
echo "  DOWNLOADING COMBINED MODEL FROM AWS"
echo "=================================================================="
echo "Instance IP: $INSTANCE_IP"
echo "Remote model directory: $REMOTE_MODEL_DIR"
echo "Local model directory: $LOCAL_MODEL_DIR"
echo

# Check if the process is still running
echo "Checking if the training process is still running..."
PROCESS_STATUS=$(ssh -i "${KEY_FILE}" ${USERNAME}@${INSTANCE_IP} "cd ~/emotion_training && if [ -f branched_regularization_sync_aug_pid.txt ]; then if ps -p \$(cat branched_regularization_sync_aug_pid.txt) > /dev/null; then echo 'Running'; else echo 'Stopped'; fi; else echo 'No PID file'; fi")

if [ "$PROCESS_STATUS" == "Running" ]; then
    echo "Training is still in progress. You may want to wait until it completes."
    echo "Use ./aws-setup/monitor_combined_model.sh to check the status."
    echo "Do you want to proceed anyway? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Download canceled."
        exit 0
    fi
fi

# Create local directory if it doesn't exist
mkdir -p "${LOCAL_MODEL_DIR}"

# Download the models
echo "Downloading models..."
scp -i "${KEY_FILE}" "${USERNAME}@${INSTANCE_IP}:${REMOTE_MODEL_DIR}/model_best.h5" "${LOCAL_MODEL_DIR}/"
scp -i "${KEY_FILE}" "${USERNAME}@${INSTANCE_IP}:${REMOTE_MODEL_DIR}/final_model.h5" "${LOCAL_MODEL_DIR}/"

# Download the training logs
echo "Downloading training logs..."
scp -i "${KEY_FILE}" "${USERNAME}@${INSTANCE_IP}:~/emotion_training/training_branched_regularization_sync_aug.log" "${LOCAL_MODEL_DIR}/"

echo
echo "Download complete!"
echo "Models and logs saved to: ${LOCAL_MODEL_DIR}"
echo "=================================================================="
