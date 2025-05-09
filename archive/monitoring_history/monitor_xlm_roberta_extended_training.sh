#!/bin/bash
set -e

SSH_KEY="/Users/patrickgloria/Downloads/gpu-key.pem"
EC2_IP=$(cat aws_instance_ip.txt)
EXP_NAME="xlm-roberta-large_extended_training"

echo "Checking if XLM-RoBERTa-large extended training (15 epochs) is running..."

# Check if the process is running
RUNNING_PROCESS=$(ssh -i "$SSH_KEY" ubuntu@$EC2_IP "ps aux | grep python | grep train_xlm_roberta_large_extended.py | grep -v grep")

if [ -z "$RUNNING_PROCESS" ]; then
    echo "WARNING: No XLM-RoBERTa extended training process is currently running!"
    echo "The training might have completed or been terminated."
    
    # Check if model exists
    MODEL_EXISTS=$(ssh -i "$SSH_KEY" ubuntu@$EC2_IP "ls -la /home/ubuntu/training_logs_humor/$EXP_NAME/final_model 2>/dev/null | wc -l")
    if [ "$MODEL_EXISTS" -gt "0" ]; then
        echo "Good news! Final model was found at /home/ubuntu/training_logs_humor/$EXP_NAME/final_model"
        echo "To download the model, run:"
        echo "  scp -r -i $SSH_KEY ubuntu@$EC2_IP:/home/ubuntu/training_logs_humor/$EXP_NAME/final_model ."
    fi
else
    echo "Found active training process:"
    echo "$RUNNING_PROCESS"
    echo ""
    echo "Starting monitoring..."
    echo "Press Ctrl+C to stop monitoring."
    echo ""

    # Check for latest validation metrics (if any)
    echo "Latest validation metrics:"
    ssh -i "$SSH_KEY" ubuntu@$EC2_IP "grep -E 'Epoch.*val_f1=' /home/ubuntu/training_logs_humor/$EXP_NAME/training.log | tail -5 || echo 'No validation metrics found yet'"
    echo ""

    # Check available checkpoints
    echo "Model checkpoints:"
    ssh -i "$SSH_KEY" ubuntu@$EC2_IP "ls -lh /home/ubuntu/training_logs_humor/$EXP_NAME/checkpoints/ 2>/dev/null || echo 'No checkpoints yet'"
    echo ""

    echo "Starting continuous log stream:"
    echo "==========================================="

    # Stream the training log continuously
    ssh -i "$SSH_KEY" ubuntu@$EC2_IP "tail -f /home/ubuntu/training_logs_humor/$EXP_NAME/training.log | grep -v 'v_num' | grep -v '/v_num' | grep -v 'version_num' | grep -v 'version_number' | grep -v 'ver_num' | grep -v 'step=' | grep -v 'step [0-9]' | grep -v 'steps=' | grep -v 'steps [0-9]' | grep -v '_step=' | grep -v 'step_[0-9]' | grep -v 'global_step' | grep -v 'batch_idx' | grep -v 'batch_index' | grep -v 'total_steps' | grep -v 'num_steps' | grep -v 'step.*:' | grep -v 'steps.*:' | grep -v 'GPU available' | grep -v 'LOCAL_RANK' | grep -v 'train.py:' | grep -v '/home/ubuntu/.local' | grep -v 'Epoch 0:' | grep -v ': \[' | grep -v 'it/s' | sed -r 's/\[[0-9;]+m//g'"
fi
