#!/bin/bash
# Deploy XLM-RoBERTa v2 with fixed manifest paths to EC2

# Get EC2 instance IP from file
EC2_IP=$(cat aws_instance_ip.txt)
SSH_KEY="/Users/patrickgloria/Downloads/gpu-key.pem"
EC2_USER="ubuntu"
EC2_TARGET_DIR="/home/ubuntu/conjunction-train"

# Make files executable
chmod +x run_xlm_roberta_v2_fixed.sh

echo "=== Deploying XLM-RoBERTa v2 with corrected dataset paths to EC2 ==="
echo "EC2 Instance IP: ${EC2_IP}"

# Upload the fixed files to EC2
echo "Uploading fixed training script and launcher..."
scp -i "${SSH_KEY}" run_xlm_roberta_v2_fixed.sh fixed_train_xlm_roberta_script_v2.py ${EC2_USER}@${EC2_IP}:${EC2_TARGET_DIR}/

# Make file executable on remote server
ssh -i "${SSH_KEY}" ${EC2_USER}@${EC2_IP} "chmod +x ${EC2_TARGET_DIR}/run_xlm_roberta_v2_fixed.sh"

# Verify dataset paths exist on EC2
echo "Verifying the ur_funny dataset paths on EC2..."
ssh -i "${SSH_KEY}" ${EC2_USER}@${EC2_IP} "cd ${EC2_TARGET_DIR} && ls -l datasets/manifests/humor/ur_funny_*_humor_cleaned.csv"

# Check for previous runs and clean up if needed
echo "Checking for previous runs and cleaning up..."
ssh -i "${SSH_KEY}" ${EC2_USER}@${EC2_IP} "pkill -f fixed_train_xlm_roberta_script_v2.py || true"
ssh -i "${SSH_KEY}" ${EC2_USER}@${EC2_IP} "rm -f ${EC2_TARGET_DIR}/xlm_roberta_v2_training.log || true"

# Upload the monitoring script as well
echo "Uploading monitoring script..."
scp -i "${SSH_KEY}" monitor_xlm_roberta_v2_training.sh ${EC2_USER}@${EC2_IP}:${EC2_TARGET_DIR}/
ssh -i "${SSH_KEY}" ${EC2_USER}@${EC2_IP} "chmod +x ${EC2_TARGET_DIR}/monitor_xlm_roberta_v2_training.sh"

# Verify GPU availability
echo "Verifying GPU availability on EC2..."
ssh -i "${SSH_KEY}" ${EC2_USER}@${EC2_IP} "nvidia-smi"

# Set up launch command
echo "Setting up launch command..."
LAUNCH_CMD="cd ${EC2_TARGET_DIR} && ./run_xlm_roberta_v2_fixed.sh > xlm_roberta_v2_training.log 2>&1 &"

# Start training
echo "Starting XLM-RoBERTa v2 training on EC2..."
ssh -i "${SSH_KEY}" ${EC2_USER}@${EC2_IP} "${LAUNCH_CMD}"
echo "Training started in background."

echo "To monitor the training, run:"
echo "  ./monitor_xlm_roberta_v2_training.sh"
echo
echo "Deployment complete!"
