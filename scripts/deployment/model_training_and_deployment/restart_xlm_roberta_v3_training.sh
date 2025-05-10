#!/bin/bash
set -e

# Get EC2 instance IP from file or environment variable
if [ -f aws_instance_ip.txt ]; then
    EC2_IP=$(cat aws_instance_ip.txt)
elif [ -n "$EC2_INSTANCE_IP" ]; then
    EC2_IP=$EC2_INSTANCE_IP
else
    echo "Error: EC2 instance IP not found. Please set EC2_INSTANCE_IP or create aws_instance_ip.txt."
    exit 1
fi

# Define EC2 username and directories
EC2_USER="ubuntu"
EC2_PROJECT_DIR="~/humor_detection"
SSH_KEY="/Users/patrickgloria/Downloads/gpu-key.pem"

echo "Restarting XLM-RoBERTa V3 training on EC2 instance at $EC2_IP..."

# Clean up the previous failed process if it exists
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" $EC2_USER@$EC2_IP "if [ -f $EC2_PROJECT_DIR/xlm_roberta_v3_training.pid ]; then 
    PID=\$(cat $EC2_PROJECT_DIR/xlm_roberta_v3_training.pid)
    if ps -p \$PID > /dev/null; then 
        echo \"Stopping previous training process (PID: \$PID)\"
        kill \$PID 2>/dev/null || true
    fi
    rm -f $EC2_PROJECT_DIR/xlm_roberta_v3_training.pid 
fi"

# Launch the training script
echo "Starting XLM-RoBERTa V3 training..."
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" $EC2_USER@$EC2_IP "cd $EC2_PROJECT_DIR && bash launch_xlm_roberta_v3.sh"

echo "XLM-RoBERTa V3 training has been restarted."
echo "Use './monitor_xlm_roberta_v3.sh' to check training progress."
