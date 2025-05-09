#!/bin/bash
# Script to monitor the LSTM attention model (no augmentation) training process

INSTANCE_IP="3.235.76.0"
KEY_FILE="aws-setup/emotion-recognition-key-fixed-20250323090016.pem"
LOG_FILE="training_lstm_attention_no_aug.log"

echo "==================================================="
echo "     MONITORING LSTM ATTENTION MODEL (NO AUG)"
echo "==================================================="
echo "Instance IP: $INSTANCE_IP"
echo "Log file: $LOG_FILE"
echo "Started at: $(date)"
echo "==================================================="

# Connect to the instance and stream the log file
ssh -i "${KEY_FILE}" -o StrictHostKeyChecking=no ec2-user@${INSTANCE_IP} "tail -f ~/emotion_training/${LOG_FILE}"
