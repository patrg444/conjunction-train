#!/bin/bash
# Download branched regularization sync aug TCN model training logs from AWS server

echo "========================================================================="
echo "  DOWNLOADING BRANCHED REGULARIZATION SYNC AUG TCN MODEL LOGS"
echo "========================================================================="

INSTANCE_IP="3.235.76.0"
USERNAME="ec2-user"
KEY_FILE="./aws-setup/emotion-recognition-key-fixed-20250323090016.pem"
REMOTE_LOG_FILE="~/emotion_training/training_branched_regularization_sync_aug_tcn.log"
LOCAL_LOG_FILE="training_branched_regularization_sync_aug_tcn.log"

if [ ! -f "$KEY_FILE" ]; then
    echo "Error: SSH key file not found: $KEY_FILE"
    echo "Please ensure the key file path is correct."
    exit 1
fi

echo "Instance IP: $INSTANCE_IP"
echo "Remote log file: $REMOTE_LOG_FILE"
echo "Local log file: $LOCAL_LOG_FILE"
echo "Downloading training logs..."

# Download the log file
scp -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP:$REMOTE_LOG_FILE" "$LOCAL_LOG_FILE"

if [ $? -eq 0 ]; then
    echo "Log file downloaded successfully."
    echo "To analyze training progress, run: ./extract_tcn_model_progress.py $LOCAL_LOG_FILE"
else
    echo "Error: Failed to download log file."
    echo "Please check your connection and try again."
    exit 1
fi

echo "========================================================================="
