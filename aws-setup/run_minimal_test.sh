#!/bin/bash
# Script to run a minimal test script to verify the environment

INSTANCE_IP="98.82.121.48"
USERNAME="ec2-user"
KEY_FILE="emotion-recognition-key-20250322082227.pem"

echo "Copying minimal test script to the instance..."
scp -i "${KEY_FILE}" -o StrictHostKeyChecking=no minimal_train.py ${USERNAME}@${INSTANCE_IP}:~/emotion_training/

echo "Running the minimal test script remotely..."
ssh -i "${KEY_FILE}" -o StrictHostKeyChecking=no ${USERNAME}@${INSTANCE_IP} << 'EOF'
cd ~/emotion_training
# First kill any running Python processes related to training
pkill -f "python.*train"

# Clear the log file
> training.log

# Run the minimal test script and log the output
python minimal_train.py > training.log 2>&1

echo "Minimal test executed. Output logged to training.log"
EOF

echo "Test script executed. Now let's check if it worked..."
echo "Checking the training log..."

# Wait a moment for the command to complete
sleep 2

# Check the log file
ssh -i "${KEY_FILE}" -o StrictHostKeyChecking=no ${USERNAME}@${INSTANCE_IP} "cat ~/emotion_training/training.log"
