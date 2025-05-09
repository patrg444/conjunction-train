#!/bin/bash
# Script to monitor the training process with the fixed normalization (no data leakage)

INSTANCE_IP="98.82.121.48"
USERNAME="ec2-user"
KEY_FILE="emotion-recognition-key-20250322082227.pem"

echo -e "\033[1;36m========================================================\033[0m"
echo -e "\033[1;36m  TRAINING WITH FIXED NORMALIZATION MONITOR (NO LEAKAGE)\033[0m"
echo -e "\033[1;36m========================================================\033[0m"
echo -e "\033[1mPress Ctrl+C to exit monitoring\033[0m"
echo ""

# Check if training is still running
running=$(ssh -i "${KEY_FILE}" -o StrictHostKeyChecking=no ${USERNAME}@${INSTANCE_IP} "ps aux | grep train_branched_no_leakage | grep -v grep | wc -l")

if [ "$running" -eq "0" ]; then
    echo -e "\033[1;31mTraining process is not running!\033[0m"
    echo "Check if training completed or encountered an error."
    echo ""
fi

# Display the log content with better formatting
echo -e "\033[1;33m=== TRAINING LOG (LATEST ENTRIES) ===\033[0m"
ssh -i "${KEY_FILE}" -o StrictHostKeyChecking=no ${USERNAME}@${INSTANCE_IP} "tail -50 ~/emotion_training/training_no_leakage.log"

echo ""
echo -e "\033[1;36m========================================================\033[0m"
echo "For more detailed logs, run:"
echo "cd aws-setup && ssh -i \"${KEY_FILE}\" ec2-user@${INSTANCE_IP} \"tail -100 ~/emotion_training/training_no_leakage.log\""
