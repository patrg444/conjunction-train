#!/bin/bash
# Script to monitor the training progress of the combined model

LOG_FILE="training_branched_regularization_sync_aug.log"
INSTANCE_IP="3.235.76.0"
USERNAME="ec2-user"
KEY_FILE="./aws-setup/emotion-recognition-key-fixed-20250323090016.pem"

echo "=================================================================="
echo "  MONITORING COMBINED MODEL TRAINING"
echo "=================================================================="
echo "Instance IP: $INSTANCE_IP"
echo "Log file: $LOG_FILE"
echo

# Check if the process is running
echo "Checking if the training process is still running..."
PROCESS_STATUS=$(ssh -i "${KEY_FILE}" ${USERNAME}@${INSTANCE_IP} "cd ~/emotion_training && if [ -f branched_regularization_sync_aug_pid.txt ]; then if ps -p \$(cat branched_regularization_sync_aug_pid.txt) > /dev/null; then echo 'Running'; else echo 'Stopped'; fi; else echo 'No PID file'; fi")

echo "Process status: $PROCESS_STATUS"
echo

# Get the last few lines of the log file to show progress
echo "Recent training progress:"
ssh -i "${KEY_FILE}" ${USERNAME}@${INSTANCE_IP} "cd ~/emotion_training && tail -n 30 ${LOG_FILE}"

echo
echo "To view more detailed logs:"
echo "ssh -i \"${KEY_FILE}\" ${USERNAME}@${INSTANCE_IP} \"cd ~/emotion_training && tail -n 100 ${LOG_FILE}\""
echo "=================================================================="
