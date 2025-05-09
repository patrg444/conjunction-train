#!/bin/bash

# Script to monitor ATTN-CRNN training progress on EC2
# Uses SSH to check log files and training status

EC2_ADDRESS="54.162.134.77"
KEY_PATH="~/Downloads/gpu-key.pem"
EMOTION_DIR="/home/ubuntu/emotion_project"
LOG_FILE="${EMOTION_DIR}/train_combined_attn_crnn.log"

echo "==================================================================="
echo "Monitoring ATTN-CRNN training progress on ${EC2_ADDRESS}"
echo "==================================================================="

# Check process status
echo "Checking training process status..."
ssh -i ${KEY_PATH} ubuntu@${EC2_ADDRESS} "ps aux | grep -i 'fixed_attn_crnn\|train_attn_crnn' | grep -v grep || echo 'No training process found'"

# Check latest log entries
echo -e "\nLatest log entries:"
echo "==================================================================="
ssh -i ${KEY_PATH} ubuntu@${EC2_ADDRESS} "tail -n 25 ${LOG_FILE}"

# Check model state (if exists)
echo -e "\nModel file status:"
echo "==================================================================="
ssh -i ${KEY_PATH} ubuntu@${EC2_ADDRESS} "ls -la ${EMOTION_DIR}/best_model.h5 || echo 'No model file found'"

echo -e "\nMonitoring complete. Run this script again to see updated progress."
