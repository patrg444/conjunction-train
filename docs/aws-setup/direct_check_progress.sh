#!/bin/bash
# Directly check training progress without menu

INSTANCE_IP="98.82.121.48"
# Pre-configured username - change this if needed
USERNAME="ec2-user"
KEY_FILE="emotion-recognition-key-20250322082227.pem"

echo "Connecting to instance to check training progress..."
ssh -i "${KEY_FILE}" -o StrictHostKeyChecking=no ${USERNAME}@${INSTANCE_IP} "tail -f ~/training.log"
