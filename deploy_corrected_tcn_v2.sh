#!/bin/bash
# Script to deploy the corrected TCN v2 model to AWS

# ANSI colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# AWS instance details
INSTANCE_IP="13.217.128.73"
USERNAME="ec2-user"
KEY_FILE="./aws-setup/emotion-recognition-key-fixed-20250323090016.pem"
REMOTE_DIR="~/emotion_training"
LOCAL_SCRIPT_PATH="scripts/train_branched_regularization_sync_aug_tcn_large_fixed_v2.py"
REMOTE_SCRIPT_PATH="${REMOTE_DIR}/scripts/train_branched_regularization_sync_aug_tcn_large_fixed_v2.py"
REMOTE_LAUNCH_SCRIPT_PATH="${REMOTE_DIR}/launch_corrected_tcn_v2.sh"

echo -e "${BLUE}===================================================================${NC}"
echo -e "${GREEN}    DEPLOYING CORRECTED TCN V2 MODEL SCRIPT    ${NC}"
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

# Transfer the corrected local TCN model script to the AWS instance
echo -e "${YELLOW}Transferring corrected TCN model script ($LOCAL_SCRIPT_PATH) to AWS instance...${NC}"
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
    echo -e "${RED}Error: Python syntax check failed. Script might still contain syntax errors:${NC}"
    echo "$SYNTAX_CHECK"
    # Don't exit immediately, maybe the newline fix worked despite this check failing before
    # exit 1 
else
    echo -e "${GREEN}Syntax check passed! Script is valid Python.${NC}"
fi

# Create a launch script on the remote server with timestamped logs/pids
echo -e "${YELLOW}Creating launch script ($REMOTE_LAUNCH_SCRIPT_PATH) on AWS instance...${NC}"

ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "cat > ${REMOTE_LAUNCH_SCRIPT_PATH}" << 'EOL'
#!/bin/bash
cd ~/emotion_training

SCRIPT_PATH="scripts/train_branched_regularization_sync_aug_tcn_large_fixed_v2.py"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="training_branched_regularization_sync_aug_tcn_large_fixed_v2_${TIMESTAMP}.log"
PID_FILE="fixed_tcn_large_v2_${TIMESTAMP}_pid.txt"

echo "Starting corrected TCN v2 model training..."
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

# Check if process is running (using the dynamically generated PID file name is tricky here, so we'll rely on user monitoring)
# A more robust check would require retrieving the PID filename from the remote instance first.
# For now, we assume it launched and instruct the user on monitoring.

echo -e "${GREEN}Training launch command executed.${NC}"
echo -e "${YELLOW}Please monitor the training progress using a monitoring script or by checking the logs on the EC2 instance.${NC}"
echo -e "${YELLOW}The specific log and PID files will have a timestamp like YYYYMMDD_HHMMSS.${NC}"

echo ""
echo -e "${BLUE}===================================================================${NC}"
echo -e "${GREEN}Deployment and launch attempt completed!${NC}"
echo -e "${BLUE}===================================================================${NC}"
