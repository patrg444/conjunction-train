#!/bin/bash
# Deploy and run the fixed training script with no data leakage using the conda environment

# Set up variables
INSTANCE_IP="98.82.121.48"
USERNAME="ec2-user"
KEY_FILE="emotion-recognition-key-20250322082227.pem"
SCRIPT_NAME="train_branched_no_leakage.py"
LOCAL_DIR="$(pwd)/.."
REMOTE_DIR="~/emotion_training"
LOG_FILE="training_no_leakage_conda.log"

echo "==================================================="
echo "  DEPLOYING FIXED MODEL TRAINING WITH CONDA (NO DATA LEAKAGE)"
echo "==================================================="
echo "Instance IP: $INSTANCE_IP"
echo "Script: $SCRIPT_NAME"
echo "Log file: $LOG_FILE"
echo

# Transfer the script
echo "Transferring fixed training script..."
scp -i "${KEY_FILE}" "${LOCAL_DIR}/scripts/${SCRIPT_NAME}" "${USERNAME}@${INSTANCE_IP}:${REMOTE_DIR}/scripts/"

# Make sure the script is executable
ssh -i "${KEY_FILE}" ${USERNAME}@${INSTANCE_IP} "chmod +x ${REMOTE_DIR}/scripts/${SCRIPT_NAME}"

# Create a run script that uses the conda environment with correct path
echo "Creating conda-aware run script on remote server..."
ssh -i "${KEY_FILE}" ${USERNAME}@${INSTANCE_IP} "cat > ${REMOTE_DIR}/run_no_leakage_conda.sh << 'EOL'
#!/bin/bash
cd ~/emotion_training

# Activate the conda environment - Using correct absolute path
source /home/ec2-user/anaconda3/etc/profile.d/conda.sh
conda activate tensorflow2_p310

# Run the fixed no data leakage script
nohup python scripts/${SCRIPT_NAME} > ${LOG_FILE} 2>&1 &
echo \$! > no_leakage_conda_pid.txt
echo \"Training process started with PID \$(cat no_leakage_conda_pid.txt)\"
echo \"Logs are being written to ${LOG_FILE}\"

# Print environment info for troubleshooting
echo \"Python version:\" >> ${LOG_FILE}
python --version >> ${LOG_FILE}
echo \"NumPy version:\" >> ${LOG_FILE}
python -c \"import numpy; print(numpy.__version__)\" >> ${LOG_FILE}
echo \"TensorFlow version:\" >> ${LOG_FILE}
python -c \"import tensorflow; print(tensorflow.__version__)\" >> ${LOG_FILE}
EOL"

# Make run script executable
ssh -i "${KEY_FILE}" ${USERNAME}@${INSTANCE_IP} "chmod +x ${REMOTE_DIR}/run_no_leakage_conda.sh"

# Run the script
echo "Starting training process on remote server with conda environment..."
ssh -i "${KEY_FILE}" ${USERNAME}@${INSTANCE_IP} "${REMOTE_DIR}/run_no_leakage_conda.sh"

echo
echo "Deployment complete!"
echo "To monitor training, run: cd aws-setup && ./monitor_no_leakage_conda.sh"
echo "==================================================="
