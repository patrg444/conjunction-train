#!/bin/bash
# Deploy and run the Python 2.7 compatible, fixed training script with no data leakage

# Set up variables
INSTANCE_IP="98.82.121.48"
USERNAME="ec2-user"
KEY_FILE="emotion-recognition-key-20250322082227.pem"
SCRIPT_NAME="train_branched_no_leakage_py2.py"
LOCAL_DIR="$(pwd)/.."
REMOTE_DIR="~/emotion_training"
LOG_FILE="training_no_leakage_py2.log"

echo "==================================================="
echo "  DEPLOYING FIXED MODEL TRAINING (PYTHON 2.7 COMPATIBLE)"
echo "==================================================="
echo "Instance IP: $INSTANCE_IP"
echo "Script: $SCRIPT_NAME"
echo "Log file: $LOG_FILE"
echo

# Transfer the script
echo "Transferring fixed Python 2.7 training script..."
scp -i "${KEY_FILE}" "${LOCAL_DIR}/scripts/${SCRIPT_NAME}" "${USERNAME}@${INSTANCE_IP}:${REMOTE_DIR}/scripts/"

# Make sure the script is executable
ssh -i "${KEY_FILE}" ${USERNAME}@${INSTANCE_IP} "chmod +x ${REMOTE_DIR}/scripts/${SCRIPT_NAME}"

# Create a run script
echo "Creating run script on remote server..."
ssh -i "${KEY_FILE}" ${USERNAME}@${INSTANCE_IP} "cat > ${REMOTE_DIR}/run_no_leakage_py2.sh << 'EOL'
#!/bin/bash
cd ~/emotion_training
nohup python scripts/${SCRIPT_NAME} > ${LOG_FILE} 2>&1 &
echo \$! > no_leakage_py2_pid.txt
echo \"Training process started with PID \$(cat no_leakage_py2_pid.txt)\"
echo \"Logs are being written to ${LOG_FILE}\"
EOL"

# Make run script executable
ssh -i "${KEY_FILE}" ${USERNAME}@${INSTANCE_IP} "chmod +x ${REMOTE_DIR}/run_no_leakage_py2.sh"

# Run the script
echo "Starting training process on remote server..."
ssh -i "${KEY_FILE}" ${USERNAME}@${INSTANCE_IP} "${REMOTE_DIR}/run_no_leakage_py2.sh"

echo
echo "Deployment complete!"
echo "To monitor training, run: cd aws-setup && ./monitor_no_leakage_py2.sh"
echo "==================================================="
