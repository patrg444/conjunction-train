#!/bin/bash
# Enhanced monitoring script specifically for the improved model with higher accuracy

INSTANCE_IP="98.82.121.48"
USERNAME="ec2-user"
KEY_FILE="emotion-recognition-key-20250322082227.pem"

echo -e "\033[1;36m========================================================\033[0m"
echo -e "\033[1;36m   IMPROVED MODEL TRAINING MONITOR (TARGET: >70%)\033[0m"
echo -e "\033[1;36m========================================================\033[0m"
echo -e "\033[1mPress Ctrl+C to exit monitoring\033[0m"
echo ""

# Check if training is still running
running=$(ssh -i "${KEY_FILE}" -o StrictHostKeyChecking=no ${USERNAME}@${INSTANCE_IP} "ps aux | grep train_improved | grep -v grep | wc -l")

if [ "$running" -eq "0" ]; then
    echo -e "\033[1;31mTraining process is not running!\033[0m"
    echo "Check if training completed or encountered an error."
    echo ""
    # Still show the log in case training completed
fi

# Display the log content with better formatting
echo -e "\033[1;33m=== TRAINING LOG (LATEST ENTRIES) ===\033[0m"
ssh -i "${KEY_FILE}" -o StrictHostKeyChecking=no ${USERNAME}@${INSTANCE_IP} "\
cat ~/emotion_training/training.log | grep -E 'IMPROVED|Starting enhanced|TensorFlow|Processing|Added|Combined|Data augmentation|Enhanced dataset|Class|Epoch|val_accuracy|loss|accuracy'"

echo ""
echo -e "\033[1;36m========================================================\033[0m"
echo "For more detailed logs, run:"
echo "cd aws-setup && ssh -i \"${KEY_FILE}\" ec2-user@${INSTANCE_IP} \"tail -50 ~/emotion_training/training.log\""
