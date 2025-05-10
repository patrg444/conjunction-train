#!/bin/bash
# Script to terminate the branched model with synchronized augmentation process on the EC2 instance

# Set up variables
INSTANCE_IP="3.235.76.0"
USERNAME="ec2-user"
KEY_FILE="../aws-setup/emotion-recognition-key-fixed-20250323090016.pem"
REMOTE_DIR="~/emotion_training"
PID_FILE="branched_sync_aug_pid.txt"

echo "==================================================="
echo "  TERMINATING BRANCHED MODEL WITH SYNCHRONIZED AUGMENTATION"
echo "==================================================="
echo "Instance IP: $INSTANCE_IP"
echo "Finding and terminating process..."

# SSH into the EC2 instance and kill the process
ssh -i "${KEY_FILE}" ${USERNAME}@${INSTANCE_IP} "cd ${REMOTE_DIR} && \
if [ -f ${PID_FILE} ]; then \
    PID=\$(cat ${PID_FILE}) && \
    echo \"Found training process with PID: \$PID\" && \
    kill -9 \$PID && \
    echo \"Process terminated successfully\" && \
    rm ${PID_FILE}; \
else \
    echo \"PID file not found. Searching for the process...\" && \
    PROCESS_PID=\$(ps aux | grep train_branched_sync_aug.py | grep -v grep | awk '{print \$2}') && \
    if [ ! -z \"\$PROCESS_PID\" ]; then \
        echo \"Found process with PID: \$PROCESS_PID\" && \
        kill -9 \$PROCESS_PID && \
        echo \"Process terminated successfully\"; \
    else \
        echo \"No running training process found\"; \
    fi \
fi"

echo "==================================================="
echo "  VERIFICATION"
echo "==================================================="
echo "Checking if process is still running..."

# Verify the process has been terminated
ssh -i "${KEY_FILE}" ${USERNAME}@${INSTANCE_IP} "ps aux | grep train_branched_sync_aug.py | grep -v grep"
if [ $? -ne 0 ]; then
    echo "Confirmation: No training process is running"
else
    echo "Warning: Process may still be running. Check manually."
fi

echo "==================================================="
echo "Task completed"
echo "==================================================="
