#!/bin/bash

# Get EC2 instance IP from file or environment variable
if [ -f aws_instance_ip.txt ]; then
    EC2_IP=$(cat aws_instance_ip.txt)
elif [ -n "$EC2_INSTANCE_IP" ]; then
    EC2_IP=$EC2_INSTANCE_IP
else
    echo "Error: EC2 instance IP not found. Please set EC2_INSTANCE_IP or create aws_instance_ip.txt."
    exit 1
fi

EC2_USER="ubuntu"
EC2_PROJECT_DIR="~/humor_detection"
SSH_KEY="/Users/patrickgloria/Downloads/gpu-key.pem"

echo "Checking XLM-RoBERTa V3 training status..."
echo "-----------------------------------------"

# Check if training process is still running
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" $EC2_USER@$EC2_IP "if [ -f $EC2_PROJECT_DIR/xlm_roberta_v3_training.pid ]; then 
    PID=\$(cat $EC2_PROJECT_DIR/xlm_roberta_v3_training.pid)
    if ps -p \$PID > /dev/null; then 
        echo \"Training is running (PID: \$PID)\"
    else 
        echo \"Training process (PID: \$PID) is not running. Check logs for completion or errors.\"
    fi
else 
    echo \"No PID file found. Training may not have been started.\"
fi"

# Show the last 50 lines of the log file
echo "-----------------------------------------"
echo "Recent training log:"
echo "-----------------------------------------"
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" $EC2_USER@$EC2_IP "tail -n 50 $EC2_PROJECT_DIR/xlm_roberta_v3_training.log"

# Check GPU usage
echo "-----------------------------------------"
echo "GPU Usage:"
echo "-----------------------------------------"
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" $EC2_USER@$EC2_IP "nvidia-smi"

# Check disk space
echo "-----------------------------------------"
echo "Disk Space:"
echo "-----------------------------------------"
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" $EC2_USER@$EC2_IP "df -h | grep -E '/$|/home'"
