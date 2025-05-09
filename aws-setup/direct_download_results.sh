#!/bin/bash
# Directly download training results without menu

INSTANCE_IP="98.82.121.48"
# Pre-configured username - change this if needed
USERNAME="ec2-user"
KEY_FILE="emotion-recognition-key-20250322082227.pem"

echo "Downloading training results..."
mkdir -p results
scp -i "${KEY_FILE}" -o StrictHostKeyChecking=no -r ${USERNAME}@${INSTANCE_IP}:~/emotion_training/models/branched_6class results/
echo "Results downloaded to results/branched_6class"
