#!/bin/bash
# Script to upload and run the Python 2.7 compatibility fixer

INSTANCE_IP="98.82.121.48"
USERNAME="ec2-user"
KEY_FILE="emotion-recognition-key-20250322082227.pem"

echo "Copying fix script to the instance..."
scp -i "${KEY_FILE}" -o StrictHostKeyChecking=no fix_train_script.py ${USERNAME}@${INSTANCE_IP}:~/emotion_training/

echo "Running the fix script remotely..."
ssh -i "${KEY_FILE}" -o StrictHostKeyChecking=no ${USERNAME}@${INSTANCE_IP} << 'EOF'
cd ~/emotion_training
# Kill any running Python processes related to training
pkill -f "python.*train"

# Clear the log file
> training.log

# Run the fix script and log the output
python fix_train_script.py >> training.log 2>&1

# Now run the fixed training script
echo "Starting fixed training script..." >> training.log
nohup python scripts/train_branched_6class_fixed.py >> training.log 2>&1 &

echo "Fixed script started. Check training.log for progress."
EOF

echo "Fix applied and training restarted."
echo "Wait a moment and then use direct_check_progress.sh to check if it's working."
