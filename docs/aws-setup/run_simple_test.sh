#!/bin/bash
# Script to run a simple test to check Python environment

INSTANCE_IP="98.82.121.48"
USERNAME="ec2-user"
KEY_FILE="emotion-recognition-key-20250322082227.pem"

echo "Copying simple test script to the instance..."
scp -i "${KEY_FILE}" -o StrictHostKeyChecking=no simple_test.py ${USERNAME}@${INSTANCE_IP}:~/emotion_training/

echo "Running the simple test script remotely..."
ssh -i "${KEY_FILE}" -o StrictHostKeyChecking=no ${USERNAME}@${INSTANCE_IP} << 'EOF'
cd ~/emotion_training
# Clear the log file
> testing.log

# Run the simple test script and log the output
python simple_test.py > testing.log 2>&1

echo "Simple test executed. Output logged to testing.log"
EOF

echo "Test script executed. Now let's check the results..."
echo "Checking the test log..."

# Wait a moment for the command to complete
sleep 2

# Check the log file
ssh -i "${KEY_FILE}" -o StrictHostKeyChecking=no ${USERNAME}@${INSTANCE_IP} "cat ~/emotion_training/testing.log"
