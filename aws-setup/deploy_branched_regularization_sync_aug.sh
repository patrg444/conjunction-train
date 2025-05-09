#!/bin/bash
# Deploy and run the combined model with L2 regularization, synchronized data augmentation, and TCN

# Set up variables
INSTANCE_IP="3.235.76.0"
USERNAME="ec2-user"
KEY_FILE="./aws-setup/emotion-recognition-key-fixed-20250323090016.pem"
SCRIPT_NAME="train_branched_regularization_sync_aug.py"
LOCAL_DIR="$(pwd)"
REMOTE_DIR="~/emotion_training"
LOG_FILE="training_branched_regularization_sync_aug_tcn.log"

echo "============================================================================="
echo "  DEPLOYING COMBINED MODEL WITH FIXED EMOTION MAPPING"
echo "  - L2 REGULARIZATION"
echo "  - SYNCHRONIZED AUGMENTATION"
echo "  - TCN ARCHITECTURE"
echo "  - SURPRISED EMOTION EXCLUDED (code '08' removed)"
echo "============================================================================="
echo "Instance IP: $INSTANCE_IP"
echo "Script: $SCRIPT_NAME"
echo "Log file: $LOG_FILE"
echo

# Transfer the script
echo "Transferring training script..."
scp -i "${KEY_FILE}" "./scripts/${SCRIPT_NAME}" "${USERNAME}@${INSTANCE_IP}:${REMOTE_DIR}/scripts/"

# Transfer the required data generator files if they don't exist on the remote server
echo "Ensuring synchronized_data_generator.py is available on the remote server..."
scp -i "${KEY_FILE}" "./scripts/synchronized_data_generator.py" "${USERNAME}@${INSTANCE_IP}:${REMOTE_DIR}/scripts/"

echo "Ensuring sequence_data_generator.py is available on the remote server..."
scp -i "${KEY_FILE}" "./scripts/sequence_data_generator.py" "${USERNAME}@${INSTANCE_IP}:${REMOTE_DIR}/scripts/"

# Make sure the script is executable
ssh -i "${KEY_FILE}" ${USERNAME}@${INSTANCE_IP} "chmod +x ${REMOTE_DIR}/scripts/${SCRIPT_NAME}"

# Create a run script that uses the system Python
echo "Creating run script on remote server..."
ssh -i "${KEY_FILE}" ${USERNAME}@${INSTANCE_IP} "cat > ${REMOTE_DIR}/run_branched_regularization_sync_aug.sh << 'EOL'
#!/bin/bash
cd ~/emotion_training

# Run the combined script with Python 3.9
nohup /usr/bin/python3.9 scripts/${SCRIPT_NAME} > ${LOG_FILE} 2>&1 &
echo \$! > branched_regularization_sync_aug_tcn_pid.txt
echo \"Training process started with PID \$(cat branched_regularization_sync_aug_tcn_pid.txt)\"
echo \"Logs are being written to ${LOG_FILE}\"

# Print environment info for troubleshooting
echo \"Python version:\" >> ${LOG_FILE}
/usr/bin/python3.9 --version >> ${LOG_FILE}
echo \"NumPy version:\" >> ${LOG_FILE}
/usr/bin/python3.9 -c \"import numpy; print(numpy.__version__)\" >> ${LOG_FILE} 2>&1 || echo \"NumPy not installed\" >> ${LOG_FILE}
echo \"TensorFlow version:\" >> ${LOG_FILE}
/usr/bin/python3.9 -c \"import tensorflow; print(tensorflow.__version__)\" >> ${LOG_FILE} 2>&1 || echo \"TensorFlow not installed\" >> ${LOG_FILE}
EOL"

# Make run script executable
ssh -i "${KEY_FILE}" ${USERNAME}@${INSTANCE_IP} "chmod +x ${REMOTE_DIR}/run_branched_regularization_sync_aug.sh"

# Run the script
echo "Starting training process on remote server..."
ssh -i "${KEY_FILE}" ${USERNAME}@${INSTANCE_IP} "${REMOTE_DIR}/run_branched_regularization_sync_aug.sh"

echo
echo "Deployment complete!"
echo "To monitor training, run: ./tcn_model_tracking.sh"
echo "============================================================================="
