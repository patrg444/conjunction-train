#!/bin/bash
# remote_train_humor_fusion_workflow.sh
# Script to SSH into EC2 instance and run the training workflow there

set -e  # Exit immediately if a command exits with a non-zero status

# EC2 instance details
EC2_KEY="$HOME/Downloads/gpu-key.pem"
EC2_HOST="ubuntu@3.80.203.65"

echo "========================================="
echo "Connecting to EC2 instance and running training workflow"
echo "========================================="

# Check that all necessary files exist locally
if [ ! -f "train_humor_fusion_workflow.sh" ]; then
  echo "Error: train_humor_fusion_workflow.sh not found locally"
  exit 1
fi

if [ ! -f "enhanced_train_distil_humor.py" ]; then
  echo "Error: enhanced_train_distil_humor.py not found locally"
  exit 1
fi

if [ ! -f "enhanced_train_distil_humor.sh" ]; then
  echo "Error: enhanced_train_distil_humor.sh not found locally"
  exit 1
fi

echo "Copying training files to EC2 instance..."
echo "1. Copying training workflow script..."
scp -i "$EC2_KEY" train_humor_fusion_workflow.sh $EC2_HOST:/home/ubuntu/conjunction-train/

echo "2. Copying enhanced_train_distil_humor.py..."
scp -i "$EC2_KEY" enhanced_train_distil_humor.py $EC2_HOST:/home/ubuntu/conjunction-train/

echo "3. Copying enhanced_train_distil_humor.sh..."
scp -i "$EC2_KEY" enhanced_train_distil_humor.sh $EC2_HOST:/home/ubuntu/conjunction-train/

# Make sure the script is executable on the EC2 instance
echo "Setting execute permissions for scripts..."
ssh -i "$EC2_KEY" $EC2_HOST "chmod +x /home/ubuntu/conjunction-train/enhanced_train_distil_humor.sh"

# Execute the training workflow on EC2
echo "Running training workflow on EC2 instance..."
ssh -i "$EC2_KEY" $EC2_HOST "cd /home/ubuntu/conjunction-train && bash train_humor_fusion_workflow.sh"
