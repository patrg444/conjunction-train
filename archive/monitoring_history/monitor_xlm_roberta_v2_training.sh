#!/bin/bash
# Monitor XLM-RoBERTa v2 training progress remotely

# Get EC2 instance IP from file (or set it manually if needed)
EC2_IP=$(cat aws_instance_ip.txt)
SSH_KEY="/Users/patrickgloria/Downloads/gpu-key.pem"
EC2_USER="ubuntu"
EC2_TARGET_DIR="/home/ubuntu/conjunction-train"
LOG_FILE="${EC2_TARGET_DIR}/xlm_roberta_v2_training.log"

echo "=== XLM-RoBERTa v2 Training Monitor ==="
echo "EC2 Instance IP: ${EC2_IP}"
echo "Monitoring log file: ${LOG_FILE}"
echo

echo "1. Checking if training is running..."
ssh -i "${SSH_KEY}" ${EC2_USER}@${EC2_IP} "ps aux | grep fixed_train_xlm_roberta_script_v2.py | grep -v grep"
RUNNING=$?

if [ $RUNNING -eq 0 ]; then
    echo "Training is currently running!"
    echo
else
    echo "Training doesn't appear to be running. Check the log file for any errors."
    echo
fi

echo "2. Displaying the last 30 lines of the log file:"
echo "-------------------------------------------------------------------------------"
ssh -i "${SSH_KEY}" ${EC2_USER}@${EC2_IP} "tail -n 30 ${LOG_FILE}"
echo "-------------------------------------------------------------------------------"

echo
echo "3. Options for monitoring:"
echo "a) To follow the log in real-time, run:"
echo "   ssh -i \"${SSH_KEY}\" ${EC2_USER}@${EC2_IP} \"tail -f ${LOG_FILE}\""
echo
echo "b) To check GPU usage, run:"
echo "   ssh -i \"${SSH_KEY}\" ${EC2_USER}@${EC2_IP} \"nvidia-smi\""
echo
echo "c) To check the best model saved so far, run:"
echo "   ssh -i \"${SSH_KEY}\" ${EC2_USER}@${EC2_IP} \"ls -la ${EC2_TARGET_DIR}/training_logs_humor/xlm-roberta-large_optimized/checkpoints/\""
echo

echo "Monitor complete!"
