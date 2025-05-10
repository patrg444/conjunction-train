#!/bin/bash
# Script to fix the encoding issue and restart training

INSTANCE_IP="98.82.121.48"
USERNAME="ec2-user"
KEY_FILE="emotion-recognition-key-20250322082227.pem"

echo "Copying fixed script to the instance..."
scp -i "${KEY_FILE}" -o StrictHostKeyChecking=no fixed_train_script.py ${USERNAME}@${INSTANCE_IP}:~/

echo "Running the fixed script remotely to patch and restart training..."
ssh -i "${KEY_FILE}" -o StrictHostKeyChecking=no ${USERNAME}@${INSTANCE_IP} << 'EOF'
cd ~/emotion_training
# First kill any running Python processes related to training
pkill -f "python.*train_branched_6class"

# Move our helper script into place
mv ~/fixed_train_script.py .

# Run the fixer script in the background
nohup python fixed_train_script.py > training.log 2>&1 &

echo "Fixed script executed. Training should restart automatically."
EOF

echo "Script executed. Training should restart with proper encoding."
echo "Wait a moment and then use direct_check_progress.sh to verify."
