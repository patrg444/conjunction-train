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
    
    # First, remove all checkpoints with F1 > 0.8 (these are considered faulty)
    echo "Removing checkpoints with F1 > 0.8 (potentially faulty runs)..."
    for f in *val_f1=*.ckpt; do
        if [[ -f "\$f" ]]; then
            # Extract F1 value from filename
            F1_VALUE=\$(echo \$f | grep -o 'val_f1=[0-9.]*' | sed 's/val_f1=//')
            
            # Compare with 0.8 (use bc for floating point comparison)
            if (( \$(echo "\$F1_VALUE > 0.8" | bc -l) )); then
                echo "Removing high F1 checkpoint: \$f (F1=\$F1_VALUE)"
                rm "\$f"
            fi
        fi
    done
    
    # Now, for checkpoints with F1 < 0.7, keep only the highest F1 checkpoint
    echo "Processing checkpoints with F1 < 0.7..."
    
    # Find the highest F1 value below 0.7
    HIGHEST_F1=0
    BEST_CHECKPOINT=""
    
    for f in *val_f1=*.ckpt; do
        if [[ -f "\$f" ]]; then
            # Extract F1 value from filename
            F1_VALUE=\$(echo \$f | grep -o 'val_f1=[0-9.]*' | sed 's/val_f1=//')
            
            # Check if below 0.7 and higher than current highest
            if (( \$(echo "\$F1_VALUE < 0.7" | bc -l) )) && (( \$(echo "\$F1_VALUE > \$HIGHEST_F1" | bc -l) )); then
                HIGHEST_F1=\$F1_VALUE
                BEST_CHECKPOINT="\$f"
            fi
        fi
    done
    
    # Keep only the highest F1 checkpoint below 0.7
    if [[ -n "\$BEST_CHECKPOINT" ]]; then
        echo "Keeping highest F1 checkpoint below 0.7: \$BEST_CHECKPOINT (F1=\$HIGHEST_F1)"
        
        for f in *val_f1=*.ckpt; do
            if [[ -f "\$f" && "\$f" != "\$BEST_CHECKPOINT" ]]; then
                # Extract F1 value from filename
                F1_VALUE=\$(echo \$f | grep -o 'val_f1=[0-9.]*' | sed 's/val_f1=//')
                
                # Remove if below 0.7 and not the best checkpoint
                if (( \$(echo "\$F1_VALUE < 0.7" | bc -l) )); then
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
