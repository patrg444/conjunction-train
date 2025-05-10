#!/bin/bash

SSH_KEY="/Users/patrickgloria/Downloads/gpu-key.pem"
EC2_IP=$(cat aws_instance_ip.txt)

echo "Checking if XLM-RoBERTa-large training is running..."

# Check if the process is running
RUNNING_PROCESS=$(ssh -i "$SSH_KEY" ubuntu@$EC2_IP "ps aux | grep python | grep train_xlm_roberta_large.py | grep -v grep")

if [ -z "$RUNNING_PROCESS" ]; then
    echo "ERROR: No XLM-RoBERTa training process is currently running!"
    echo "Please start training first using ./start_improved_xlm_roberta_training.sh"
    echo "or any other training initialization script."
    exit 1
fi

echo "Found active training process:"
echo "$RUNNING_PROCESS"
echo ""
echo "Starting monitoring..."
echo "Press Ctrl+C to stop monitoring."
echo ""

# Check for latest validation metrics (if any)
echo "Latest validation metrics:"
ssh -i "$SSH_KEY" ubuntu@$EC2_IP "grep -E 'Epoch.*val_f1=' /home/ubuntu/training_logs_humor/xlm-roberta-large/training.log | tail -5"
echo ""

# Check available checkpoints
echo "Model checkpoints:"
ssh -i "$SSH_KEY" ubuntu@$EC2_IP "ls -lh /home/ubuntu/training_logs_humor/xlm-roberta-large/checkpoints/ 2>/dev/null || echo 'No checkpoints yet'"
echo ""

echo "Starting continuous log stream (enhanced filtering):"
echo "==========================================="

# Stream the training log continuously with enhanced filtering
# Filter out step values, v_num, and other verbose output for cleaner monitoring
