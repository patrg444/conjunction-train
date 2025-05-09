#!/bin/bash
# Script to monitor the dynamic padding training progress

INSTANCE_IP="98.82.121.48"
USERNAME="ec2-user"
KEY_FILE="emotion-recognition-key-20250322082227.pem"

echo -e "\033[1;36m========================================================\033[0m"
echo -e "\033[1;36m  DYNAMIC SEQUENCE PADDING TRAINING MONITOR\033[0m"
echo -e "\033[1;36m========================================================\033[0m"
echo -e "\033[1mPress Ctrl+C to exit monitoring\033[0m"
echo ""

# Check if training is still running
running=$(ssh -i "${KEY_FILE}" -o StrictHostKeyChecking=no ${USERNAME}@${INSTANCE_IP} "ps aux | grep train_branched_with_dynamic_padding | grep -v grep | wc -l")

if [ "$running" -eq "0" ]; then
    echo -e "\033[1;31mDynamic padding training process is not running!\033[0m"
    echo "Check if training completed or encountered an error."
    echo ""
fi

# Display the log content with better formatting
echo -e "\033[1;33m=== TRAINING LOG (LATEST ENTRIES) ===\033[0m"
ssh -i "${KEY_FILE}" -o StrictHostKeyChecking=no ${USERNAME}@${INSTANCE_IP} "\
cat ~/emotion_training/training_dynamic.log | grep -E 'DYNAMIC|Starting|TensorFlow|Processing|Added|Combined|Data augmentation|Enhanced dataset|Class|Epoch|val_accuracy|loss|accuracy|sequence length|PADDING'"

echo ""
echo -e "\033[1;36m========================================================\033[0m"
echo "For more detailed logs, run:"
echo "cd aws-setup && ssh -i \"${KEY_FILE}\" ec2-user@${INSTANCE_IP} \"tail -50 ~/emotion_training/training_dynamic.log\""
