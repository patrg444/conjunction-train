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

echo "Connecting to EC2 instance at $EC2_IP to clean up checkpoints based on F1 scores..."

# Connect to EC2 and run cleanup commands
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" $EC2_USER@$EC2_IP << EOF
    echo "Current disk usage:"
    df -h /
    
    echo "Checkpoint directory size before cleanup:"
    du -sh $CHECKPOINTS_DIR
    
    echo "Listing checkpoint files..."
    cd $CHECKPOINTS_DIR
    ls -la
    
    echo "Analyzing checkpoints based on F1 scores..."

    # Remove checkpoint with F1 = 1.0000 (definitely faulty)
    echo "Removing checkpoint with unrealistic F1 of 1.0 (definitely faulty)..."
    if ls *val_f1=1.0000* >/dev/null 2>&1; then
        for f in *val_f1=1.0000*; do
            echo "Removing unrealistic F1 checkpoint: \$f"
            rm "\$f"
        done
    else
        echo "No checkpoints with F1=1.0000 found."
    fi

    # Get the best checkpoint with F1 < 0.7 (legitimate training runs)
    echo "Identifying best checkpoint with F1 < 0.7..."
    BEST_CHECKPOINT=""
    BEST_F1="0.0"

    for f in *val_f1=*.ckpt; do
        if [ -f "\$f" ]; then
            # Extract F1 value from filename
            F1_VALUE=\$(echo \$f | grep -o 'val_f1=[0-9.]*' | sed 's/val_f1=//')
            
            # Round to 4 decimal places to avoid precision issues
            F1_ROUNDED=\$(printf "%.4f" \$F1_VALUE)
            
            # Keep checkpoints with F1 < 0.7
            if [ \$(echo "\$F1_ROUNDED < 0.7000" | bc -l) -eq 1 ]; then
                if [ \$(echo "\$F1_ROUNDED > \$BEST_F1" | bc -l) -eq 1 ]; then
                    BEST_F1="\$F1_ROUNDED"
                    BEST_CHECKPOINT="\$f"
                fi
            fi
        fi
    done

    if [ -n "\$BEST_CHECKPOINT" ]; then
        echo "Best checkpoint with F1 < 0.7: \$BEST_CHECKPOINT (F1=\$BEST_F1)"
        
        # Delete other checkpoints with F1 < 0.7
        for f in *val_f1=*.ckpt; do
            if [ -f "\$f" ] && [ "\$f" != "\$BEST_CHECKPOINT" ]; then
                F1_VALUE=\$(echo \$f | grep -o 'val_f1=[0-9.]*' | sed 's/val_f1=//')
                F1_ROUNDED=\$(printf "%.4f" \$F1_VALUE)
                
                if [ \$(echo "\$F1_ROUNDED < 0.7000" | bc -l) -eq 1 ]; then
                    echo "Removing lower F1 checkpoint: \$f (F1=\$F1_VALUE)"
                    rm "\$f"
                fi
            fi
        done
    else
        echo "No checkpoints with F1 < 0.7 found."
    fi
    
    echo "Remaining checkpoints after cleanup:"
    ls -la
    
    echo "Checkpoint directory size after cleanup:"
    du -sh $CHECKPOINTS_DIR
    
    echo "Current disk usage after cleanup:"
    df -h /
EOF

echo "Checkpoint cleanup by F1 score completed."
