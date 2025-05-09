#!/bin/bash
set -e

# Get EC2 instance IP from file
if [ -f aws_instance_ip.txt ]; then
    EC2_IP=$(cat aws_instance_ip.txt)
else
    echo "Error: EC2 instance IP not found. Please create aws_instance_ip.txt."
    exit 1
fi

# Define EC2 username and the SSH key
EC2_USER="ubuntu"
SSH_KEY="/Users/patrickgloria/Downloads/gpu-key.pem"

# Path to the checkpoints directory
CHECKPOINTS_DIR="/home/ubuntu/humor_detection/training_logs_humor/xlm-roberta-large_v3_optimized/checkpoints"

echo "Connecting to EC2 instance at $EC2_IP to clean up checkpoints..."

# Connect to EC2 and run cleanup commands
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" $EC2_USER@$EC2_IP << EOF
    echo "Current disk usage:"
    df -h /
    
    echo "Checkpoint directory size before cleanup:"
    du -sh $CHECKPOINTS_DIR
    
    echo "Listing checkpoint files..."
    cd $CHECKPOINTS_DIR
    
    # Get the last checkpoint (best model)
    BEST_MODEL=\$(ls -t | grep "best" | head -n 1)
    
    # Get the most recent checkpoint by time
    LAST_CHECKPOINT=\$(ls -t epoch*.ckpt | head -n 1)
    
    # Keep the best model and the most recent checkpoint
    echo "Keeping best model: \$BEST_MODEL"
    echo "Keeping most recent checkpoint: \$LAST_CHECKPOINT"
    
    # Delete all other checkpoints
    echo "Deleting other checkpoints..."
    for f in *.ckpt; do
        if [[ "\$f" != "\$BEST_MODEL" && "\$f" != "\$LAST_CHECKPOINT" ]]; then
            echo "Removing \$f"
            rm "\$f"
        fi
    done
    
    echo "Checkpoint directory size after cleanup:"
    du -sh $CHECKPOINTS_DIR
    
    echo "Current disk usage after cleanup:"
    df -h /
EOF

echo "Checkpoint cleanup completed."
