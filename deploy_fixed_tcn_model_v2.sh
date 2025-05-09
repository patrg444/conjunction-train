#!/bin/bash
# Improved script to deploy the fixed TCN model to AWS with better error handling

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
SCRIPT_PATH="${REMOTE_DIR}/scripts/train_branched_regularization_sync_aug_tcn_large_fixed_complete.py"

echo -e "${BLUE}===================================================================${NC}"
echo -e "${GREEN}    DEPLOYING FIXED TCN MODEL WITH ENHANCED DEPLOYMENT SCRIPT    ${NC}"
echo -e "${BLUE}===================================================================${NC}"
echo ""

# Check if key file exists
if [ ! -f "$KEY_FILE" ]; then
    echo -e "${RED}Error: SSH key file not found: $KEY_FILE${NC}"
    echo "Please ensure the key file path is correct."
    exit 1
fi

# Transfer the local fixed TCN model script to the AWS instance
echo -e "${YELLOW}Transferring fixed TCN model script to AWS instance...${NC}"
scp -i "$KEY_FILE" scripts/train_branched_regularization_sync_aug_tcn_large_fixed_complete.py "$USERNAME@$INSTANCE_IP:${SCRIPT_PATH}"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to transfer the script to AWS instance.${NC}"
    exit 1
fi

echo -e "${GREEN}Script transferred successfully!${NC}"

# Set execute permissions on the remote script
echo -e "${YELLOW}Setting execute permissions on the script...${NC}"
ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "chmod +x ${SCRIPT_PATH}"

# Check if script is valid Python syntax
echo -e "${YELLOW}Checking script for Python syntax errors...${NC}"
SYNTAX_CHECK=$(ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "cd ${REMOTE_DIR} && python3 -m py_compile ${SCRIPT_PATH} 2>&1")

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Python syntax check failed. Script contains syntax errors:${NC}"
    echo "$SYNTAX_CHECK"
    echo -e "${YELLOW}Attempting to fix common syntax issues...${NC}"
    
    # Use dos2unix to fix line endings if available
    ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "which dos2unix > /dev/null && dos2unix ${SCRIPT_PATH} || echo 'dos2unix not available'"
    
    # Run syntax check again
    SYNTAX_CHECK=$(ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "cd ${REMOTE_DIR} && python3 -m py_compile ${SCRIPT_PATH} 2>&1")
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Failed to fix syntax issues automatically. Manual intervention required.${NC}"
        echo "$SYNTAX_CHECK"
        exit 1
    else
        echo -e "${GREEN}Syntax issues fixed successfully!${NC}"
    fi
else
    echo -e "${GREEN}Syntax check passed! Script is valid Python.${NC}"
fi

# Create a launch script on the remote server
echo -e "${YELLOW}Creating launch script on AWS instance...${NC}"

ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "cat > ${REMOTE_DIR}/launch_fixed_tcn_v2.sh" << 'EOL'
#!/bin/bash
cd ~/emotion_training

SCRIPT_PATH="scripts/train_branched_regularization_sync_aug_tcn_large_fixed_complete.py"

echo "Starting fixed TCN model training..."
nohup python3 $SCRIPT_PATH > training_branched_regularization_sync_aug_tcn_large_fixed_complete.log 2>&1 &

# Save PID for monitoring
echo $! > fixed_tcn_large_pid.txt
echo "Training process started with PID: $(cat fixed_tcn_large_pid.txt)"
echo "Logs are being written to: training_branched_regularization_sync_aug_tcn_large_fixed_complete.log"

# Display Python version info for debugging
echo "Python and TensorFlow versions:"
python3 --version
python3 -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
EOL

# Set execute permissions on the launch script
ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "chmod +x ${REMOTE_DIR}/launch_fixed_tcn_v2.sh"

echo -e "${GREEN}Launch script created successfully!${NC}"

# Ask if the user wants to launch the training
echo -e "${YELLOW}The fixed TCN model script and launch script have been created on the AWS instance.${NC}"
read -p "Would you like to launch the training now? (y/n): " LAUNCH_CHOICE

if [[ "$LAUNCH_CHOICE" == "y" || "$LAUNCH_CHOICE" == "Y" ]]; then
    echo -e "${YELLOW}Launching training process on AWS instance...${NC}"
    ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "${REMOTE_DIR}/launch_fixed_tcn_v2.sh"
    
    # Wait a few seconds for the process to start
    sleep 5
    
    # Check if process is running
    PID_CHECK=$(ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "cat ${REMOTE_DIR}/fixed_tcn_large_pid.txt 2>/dev/null")
    if [[ -n "$PID_CHECK" ]]; then
        PROCESS_CHECK=$(ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "ps -p $PID_CHECK -o comm= 2>/dev/null")
        if [[ -n "$PROCESS_CHECK" ]]; then
            echo -e "${GREEN}Training launched successfully with PID: $PID_CHECK!${NC}"
        else
            echo -e "${RED}Training process started but may have terminated. Check logs for details.${NC}"
        fi
    else
        echo -e "${RED}Failed to get process ID. Training may not have started properly.${NC}"
    fi
    
    echo -e "${YELLOW}You can monitor the training using:${NC}"
    echo -e "  ./continuous_tcn_monitoring.sh"
    echo -e "  ./continuous_tcn_monitoring_crossplatform.sh"
else
    echo -e "${YELLOW}Training not launched. You can manually launch it later with:${NC}"
    echo -e "  ssh -i \"$KEY_FILE\" \"$USERNAME@$INSTANCE_IP\" \"${REMOTE_DIR}/launch_fixed_tcn_v2.sh\""
fi

echo ""
echo -e "${BLUE}===================================================================${NC}"
echo -e "${GREEN}Deployment completed!${NC}"
echo -e "${BLUE}===================================================================${NC}"
