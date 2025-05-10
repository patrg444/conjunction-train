#!/bin/bash
# Script to continuously monitor training progress and push updates to terminal

INSTANCE_IP="98.82.121.48"
USERNAME="ec2-user"
KEY_FILE="emotion-recognition-key-20250322082227.pem"

echo "Starting live training monitor..."
echo "Press Ctrl+C to exit the monitoring"
echo "===================================="

# SSH to the instance and continuously stream the training log
# Use grep to highlight important info like epochs and validation results
ssh -i "${KEY_FILE}" -o StrictHostKeyChecking=no ${USERNAME}@${INSTANCE_IP} "\
    tail -f ~/emotion_training/training.log | \
    grep --color=always -E 'Epoch [0-9]+/|val_loss|val_accuracy|error|Added|Combined|Processing|loss:|accuracy:'"

# This command will run until manually terminated with Ctrl+C
