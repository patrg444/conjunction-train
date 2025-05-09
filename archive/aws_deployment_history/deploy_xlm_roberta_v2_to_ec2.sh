#!/bin/bash
# Deploy XLM-RoBERTa v2 to EC2 with correct dataset paths

# Get EC2 instance IP from file (or set it manually if needed)
EC2_IP=$(cat aws_instance_ip.txt)
SSH_KEY="/Users/patrickgloria/Downloads/gpu-key.pem"
EC2_USER="ubuntu"
EC2_TARGET_DIR="/home/ubuntu/conjunction-train"

# Make files executable
chmod +x run_xlm_roberta_v2_fixed.sh

echo "=== Deploying XLM-RoBERTa v2 with correct dataset paths to EC2 ==="
echo "EC2 Instance IP: ${EC2_IP}"

# Upload the fixed files to EC2
echo "Uploading fixed training script and launcher..."
scp -i "${SSH_KEY}" run_xlm_roberta_v2_fixed.sh fixed_train_xlm_roberta_script_v2.py ${EC2_USER}@${EC2_IP}:${EC2_TARGET_DIR}/

# Make file executable on remote server
ssh -i "${SSH_KEY}" ${EC2_USER}@${EC2_IP} "chmod +x ${EC2_TARGET_DIR}/run_xlm_roberta_v2_fixed.sh"

# Verify UR-Funny dataset exists and print paths if found
echo "Verifying dataset paths on EC2..."
ssh -i "${SSH_KEY}" ${EC2_USER}@${EC2_IP} "cd ${EC2_TARGET_DIR} && find . -name 'ur_funny_*_humor_cleaned.csv'"

# Verify GPU availability
echo "Checking GPU availability on EC2..."
ssh -i "${SSH_KEY}" ${EC2_USER}@${EC2_IP} "nvidia-smi"

# Set up launch command
echo "Setting up launch command..."
LAUNCH_CMD="cd ${EC2_TARGET_DIR} && ./run_xlm_roberta_v2_fixed.sh > xlm_roberta_v2_training.log 2>&1 &"

# Ask if user wants to start training
read -p "Start XLM-RoBERTa v2 training on EC2? (y/n): " START_TRAINING
if [[ $START_TRAINING == "y" || $START_TRAINING == "Y" ]]; then
    echo "Starting training on EC2..."
    ssh -i "${SSH_KEY}" ${EC2_USER}@${EC2_IP} "${LAUNCH_CMD}"
    echo "Training started in background. Monitor with:"
    echo "ssh -i \"${SSH_KEY}\" ${EC2_USER}@${EC2_IP} \"tail -f ${EC2_TARGET_DIR}/xlm_roberta_v2_training.log\""
else
    echo "Training not started. You can start it manually with:"
    echo "ssh -i \"${SSH_KEY}\" ${EC2_USER}@${EC2_IP} \"${LAUNCH_CMD}\""
fi

echo "Deployment complete!"
