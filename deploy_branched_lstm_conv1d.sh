#!/bin/bash
# Script to deploy the Branched LSTM/Conv1D model to AWS

# ANSI colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# AWS instance details
INSTANCE_IP="18.206.48.166"
USERNAME="ec2-user"
KEY_FILE="./aws-setup/emotion-recognition-key-fixed-20250323090016.pem"
REMOTE_DIR="~/emotion_training"
LOCAL_SCRIPT_PATH="scripts/train_branched_lstm_conv1d.py" # Path to the new script
REMOTE_SCRIPT_PATH="${REMOTE_DIR}/scripts/train_branched_lstm_conv1d.py" # Remote path for the new script
REMOTE_LAUNCH_SCRIPT_PATH="${REMOTE_DIR}/launch_branched_lstm_conv1d.sh" # New launch script name

echo -e "${BLUE}===================================================================${NC}"
echo -e "${GREEN}    DEPLOYING BRANCHED LSTM/Conv1D MODEL SCRIPT    ${NC}"
echo -e "${BLUE}===================================================================${NC}"
echo ""

# Check if key file exists
if [ ! -f "$KEY_FILE" ]; then
    echo -e "${RED}Error: SSH key file not found: $KEY_FILE${NC}"
    echo "Please ensure the key file path is correct."
    exit 1
fi

# Check if local script exists
if [ ! -f "$LOCAL_SCRIPT_PATH" ]; then
    echo -e "${RED}Error: Local Python script not found: $LOCAL_SCRIPT_PATH${NC}"
    exit 1
fi

# Transfer the new local script to the AWS instance
echo -e "${YELLOW}Transferring Branched LSTM/Conv1D script ($LOCAL_SCRIPT_PATH) to AWS instance...${NC}"
scp -i "$KEY_FILE" "$LOCAL_SCRIPT_PATH" "$USERNAME@$INSTANCE_IP:${REMOTE_SCRIPT_PATH}"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to transfer the script to AWS instance.${NC}"
    exit 1
fi

echo -e "${GREEN}Script transferred successfully!${NC}"

# Set execute permissions on the remote script
echo -e "${YELLOW}Setting execute permissions on the script...${NC}"
ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "chmod +x ${REMOTE_SCRIPT_PATH}"

# Check if script is valid Python syntax
echo -e "${YELLOW}Checking script for Python syntax errors...${NC}"
SYNTAX_CHECK=$(ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "cd ${REMOTE_DIR} && python3 -m py_compile ${REMOTE_SCRIPT_PATH} 2>&1")

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Python syntax check failed. Script might contain syntax errors:${NC}"
    echo "$SYNTAX_CHECK"
    # Consider exiting if syntax check fails
    # exit 1 
else
    echo -e "${GREEN}Syntax check passed! Script is valid Python.${NC}"
fi

# Create a launch script on the remote server with timestamped logs/pids for this model type
echo -e "${YELLOW}Creating launch script ($REMOTE_LAUNCH_SCRIPT_PATH) on AWS instance...${NC}"

ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "cat > ${REMOTE_LAUNCH_SCRIPT_PATH}" << 'EOL'
#!/bin/bash
cd ~/emotion_training

SCRIPT_PATH="scripts/train_branched_lstm_conv1d.py" # Use the new script
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="training_branched_lstm_conv1d_${TIMESTAMP}.log" # New log file pattern
PID_FILE="branched_lstm_conv1d_${TIMESTAMP}_pid.txt" # New PID file pattern

echo "Starting Branched LSTM/Conv1D model training..."
nohup python3 $SCRIPT_PATH > $LOG_FILE 2>&1 &

# Save PID for monitoring
echo $! > $PID_FILE
echo "Training process started with PID: $(cat $PID_FILE)"
echo "Logs are being written to: $LOG_FILE"

# Display Python version info for debugging
echo "Python and TensorFlow versions:"
python3 --version
python3 -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
EOL

# Set execute permissions on the launch script
ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "chmod +x ${REMOTE_LAUNCH_SCRIPT_PATH}"

echo -e "${GREEN}Launch script created successfully!${NC}"

# Launch the training directly
echo -e "${YELLOW}Launching training process on AWS instance...${NC}"
ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "${REMOTE_LAUNCH_SCRIPT_PATH}"

# Wait a few seconds for the process to start
sleep 5

echo -e "${GREEN}Training launch command executed.${NC}"
echo -e "${YELLOW}Please monitor the training progress using a specific monitoring script or by checking the logs on the EC2 instance.${NC}"
echo -e "${YELLOW}The specific log and PID files will have a timestamp like YYYYMMDD_HHMMSS and be named like 'training_branched_lstm_conv1d_...' and 'branched_lstm_conv1d_...'.${NC}"

echo ""
echo -e "${BLUE}===================================================================${NC}"
echo -e "${GREEN}Deployment and launch attempt completed!${NC}"
echo -e "${BLUE}===================================================================${NC}"
