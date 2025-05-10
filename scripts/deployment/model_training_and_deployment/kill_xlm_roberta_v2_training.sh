#!/bin/bash
# Script to stop any running XLM-RoBERTa v2 training processes on EC2

# EC2 instance details
EC2_HOST="ubuntu@3.80.203.65"
KEY_PATH="/Users/patrickgloria/Downloads/gpu-key.pem"

# Check if key file exists
if [ ! -f "$KEY_PATH" ]; then
    echo "Error: SSH key file not found at $KEY_PATH"
    exit 1
fi

echo "=== Stopping XLM-RoBERTa v2 Training ==="
echo "EC2 Instance: $EC2_HOST"
echo "Timestamp: $(date)"

# First check if any process is running
echo "Checking for running XLM-RoBERTa processes..."
ssh -i "$KEY_PATH" $EC2_HOST "ps aux | grep 'python.*xlm-roberta.*' | grep -v grep"

# Stop the processes
echo "Stopping any running XLM-RoBERTa processes..."
ssh -i "$KEY_PATH" $EC2_HOST "pkill -f 'python.*xlm-roberta.*' || echo 'No running XLM-RoBERTa processes found'"

# Check if termination was successful
echo "Verifying termination..."
sleep 2
ssh -i "$KEY_PATH" $EC2_HOST "ps aux | grep 'python.*xlm-roberta.*' | grep -v grep"

if [ $? -eq 1 ]; then
    echo "SUCCESS: All XLM-RoBERTa training processes have been terminated."
else
    echo "WARNING: Some XLM-RoBERTa processes may still be running. Forcefully terminating..."
    ssh -i "$KEY_PATH" $EC2_HOST "pkill -9 -f 'python.*xlm-roberta.*' || echo 'Force kill not needed'"
    
    # Final verification
    sleep 2
    ssh -i "$KEY_PATH" $EC2_HOST "ps aux | grep 'python.*xlm-roberta.*' | grep -v grep"
    
    if [ $? -eq 1 ]; then
        echo "SUCCESS: All XLM-RoBERTa training processes have been forcefully terminated."
    else
        echo "CRITICAL ERROR: Unable to terminate some XLM-RoBERTa processes. Manual intervention required."
        exit 1
    fi
fi

echo "Preserving any valuable model checkpoints..."
ssh -i "$KEY_PATH" $EC2_HOST "if [ -d 'training_logs_humor/xlm-roberta-large_optimized/checkpoints' ]; then cp -r training_logs_humor/xlm-roberta-large_optimized/checkpoints training_logs_humor/xlm-roberta-large_optimized/checkpoints_backup_$(date +%Y%m%d_%H%M%S); fi"

echo "All XLM-RoBERTa v2 training processes have been stopped."
echo "You can now safely deploy the XLM-RoBERTa v3 training."
echo ""
echo "To deploy XLM-RoBERTa v3 training, run:"
echo "./deploy_xlm_roberta_v3.sh"
