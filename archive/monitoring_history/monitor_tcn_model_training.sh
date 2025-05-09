#!/bin/bash
# Continuously stream logs for branched regularization sync aug TCN model

echo "========================================================================="
echo "  MONITORING BRANCHED REGULARIZATION SYNC AUG TCN MODEL TRAINING"
echo "========================================================================="

INSTANCE_IP="3.235.76.0"
USERNAME="ec2-user"
KEY_FILE="./aws-setup/emotion-recognition-key-fixed-20250323090016.pem"
LOG_FILE="training_branched_regularization_sync_aug_tcn_large.log"

if [ ! -f "$KEY_FILE" ]; then
    echo "Error: SSH key file not found: $KEY_FILE"
    echo "Please ensure the key file path is correct."
    exit 1
fi

echo "Instance IP: $INSTANCE_IP"
echo "Log file: $LOG_FILE"
echo "Starting continuous monitoring... (Press Ctrl+C to stop)"
echo

# Use SSH to continuously monitor the log file
ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "tail -f ~/emotion_training/$LOG_FILE"
