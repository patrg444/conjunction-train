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
ssh -i "$SSH_KEY" ubuntu@$EC2_IP "tail -f /home/ubuntu/training_logs_humor/xlm-roberta-large/training.log | grep -v 'v_num' | grep -v '/v_num' | grep -v 'version_num' | grep -v 'version_number' | grep -v 'ver_num' | grep -v 'step=' | grep -v 'step [0-9]' | grep -v 'steps=' | grep -v 'steps [0-9]' | grep -v '_step=' | grep -v 'step_[0-9]' | grep -v 'global_step' | grep -v 'batch_idx' | grep -v 'batch_index' | grep -v 'total_steps' | grep -v 'num_steps' | grep -v 'step.*:' | grep -v 'steps.*:' | grep -v 'GPU available' | grep -v 'LOCAL_RANK' | grep -v 'train.py:' | grep -v '/home/ubuntu/.local' | grep -v 'Epoch 0:' | grep -v ': \[' | grep -v 'it/s' | sed -r 's/\[[0-9;]+m//g'"
