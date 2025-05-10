#!/bin/bash
# Directly upload fixed script and restart training

INSTANCE_IP="98.82.121.48"
# Pre-configured username - change this if needed
USERNAME="ec2-user"
KEY_FILE="emotion-recognition-key-20250322082227.pem"

echo "Copying fixed train_branched_6class.py with UTF-8 encoding declaration..."
scp -i "${KEY_FILE}" -o StrictHostKeyChecking=no ../scripts/train_branched_6class.py ${USERNAME}@${INSTANCE_IP}:~/emotion_training/scripts/

echo "Restarting training..."
ssh -i "${KEY_FILE}" -o StrictHostKeyChecking=no ${USERNAME}@${INSTANCE_IP} "cd ~/emotion_training && nohup python scripts/train_branched_6class.py > training.log 2>&1 &"
echo "Training restarted. Use direct_check_progress.sh to monitor."
