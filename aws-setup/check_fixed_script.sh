#!/bin/bash
# Script to check if the fixed script is running

INSTANCE_IP="98.82.121.48"
USERNAME="ec2-user"
KEY_FILE="emotion-recognition-key-20250322082227.pem"

echo "Checking if fixed script exists and training is running..."
ssh -i "${KEY_FILE}" -o StrictHostKeyChecking=no ${USERNAME}@${INSTANCE_IP} << 'EOF'
cd ~/emotion_training

# Check if our fixed script exists
echo "Looking for fixed script:"
ls -la scripts/train_branched_6class_py2.py

# Try to run the fixed script directly to see if it works
echo -e "\nTrying to run the fixed script manually..."
python scripts/train_branched_6class_py2.py > test_output.log 2>&1 &
PID=$!
sleep 2
kill $PID

# Check the output
echo -e "\nInitial output from test run:"
head -20 test_output.log

# Check what's in the training log
echo -e "\nCurrent training log content:"
cat training.log
EOF

echo "Check complete."
