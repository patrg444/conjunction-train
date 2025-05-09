#!/bin/bash
# Deploy and run the Reinforcement Learning-based training script
# This script uses the Conv1D architecture but adds RL for frame selection
# Designed to run alongside existing training jobs

# Set up variables
INSTANCE_IP="3.235.76.0"
USERNAME="ec2-user"
KEY_FILE="aws-setup/emotion-recognition-key-fixed-20250323090016.pem"
SCRIPT_NAME="train_branched_rl.py"
LOCAL_DIR="$(pwd)"
REMOTE_DIR="~/emotion_training"
LOG_FILE="training_rl_model.log"

echo "==================================================="
echo "  RUNNING RL-ENHANCED EMOTION RECOGNITION MODEL"
echo "==================================================="
echo "Instance IP: $INSTANCE_IP"
echo "Script: $SCRIPT_NAME"
echo "Log file: $LOG_FILE"
echo

# Transfer the script
echo "Transferring RL model training script..."
scp -i "${KEY_FILE}" "${LOCAL_DIR}/scripts/${SCRIPT_NAME}" "${USERNAME}@${INSTANCE_IP}:${REMOTE_DIR}/scripts/"

# Make sure the script is executable
ssh -i "${KEY_FILE}" ${USERNAME}@${INSTANCE_IP} "chmod +x ${REMOTE_DIR}/scripts/${SCRIPT_NAME}"

# Check free CPU resources
echo "Checking CPU usage on EC2 instance..."
ssh -i "${KEY_FILE}" ${USERNAME}@${INSTANCE_IP} "top -b -n 1 | head -5"

# Create a run script
echo "Creating run script on remote server..."
ssh -i "${KEY_FILE}" ${USERNAME}@${INSTANCE_IP} "cat > ${REMOTE_DIR}/run_rl_model.sh << 'EOL'
#!/bin/bash
cd ~/emotion_training

# Set environment variables to use fewer CPU cores to avoid conflicts
# Give this model 24 cores, leaving cores for the other training jobs
export TF_NUM_INTEROP_THREADS=24
export TF_NUM_INTRAOP_THREADS=24
export OMP_NUM_THREADS=24

# Create directory for model if it doesn't exist
mkdir -p models/rl_frame_selection

# Run the script with reinforcement learning
echo \"Starting RL model training process...\"
nohup python3 scripts/${SCRIPT_NAME} > ${LOG_FILE} 2>&1 &

# Store PID for monitoring
echo \$! > rl_model_pid.txt
echo \"Training process started with PID \$(cat rl_model_pid.txt)\"
echo \"Logs are being written to ${LOG_FILE}\"
EOL"

# Make run script executable
ssh -i "${KEY_FILE}" ${USERNAME}@${INSTANCE_IP} "chmod +x ${REMOTE_DIR}/run_rl_model.sh"

# Run the script
echo "Starting RL model training process on remote server..."
ssh -i "${KEY_FILE}" ${USERNAME}@${INSTANCE_IP} "${REMOTE_DIR}/run_rl_model.sh"

# Create a monitoring script
cat > aws-setup/monitor_rl_model.sh << 'EOL'
#!/bin/bash
# Script to monitor the RL model training process

INSTANCE_IP="3.235.76.0"
KEY_FILE="aws-setup/emotion-recognition-key-fixed-20250323090016.pem"
LOG_FILE="training_rl_model.log"

echo "==================================================="
echo "     MONITORING RL MODEL TRAINING PROCESS"
echo "==================================================="
echo "Instance IP: $INSTANCE_IP"
echo "Log file: $LOG_FILE"
echo "Started at: $(date)"
echo "==================================================="

# Connect to the instance and stream the log file
ssh -i "${KEY_FILE}" -o StrictHostKeyChecking=no ec2-user@${INSTANCE_IP} "tail -f ~/emotion_training/${LOG_FILE}"
EOL

chmod +x aws-setup/monitor_rl_model.sh

echo
echo "RL model deployment complete!"
echo "To monitor training, run: ./aws-setup/monitor_rl_model.sh"
echo "==================================================="
