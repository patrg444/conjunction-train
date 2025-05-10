#!/bin/bash
# Script to stop the running TCN large model training process on AWS

# Instance details (from monitor_tcn_model_training.sh)
INSTANCE_IP="3.235.76.0"
USERNAME="ec2-user"
KEY_FILE="./aws-setup/emotion-recognition-key-fixed-20250323090016.pem"

# ANSI colors for readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================================================${NC}"
echo -e "${YELLOW}    STOPPING TCN LARGE MODEL TRAINING PROCESS    ${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "${YELLOW}Instance IP:${NC} $INSTANCE_IP"
echo -e "${BLUE}=================================================================${NC}"
echo ""

# Check if SSH key file exists
if [ ! -f "$KEY_FILE" ]; then
    echo -e "${RED}Error: SSH key file not found: $KEY_FILE${NC}"
    echo "Please ensure the key file path is correct."
    exit 1
fi

echo -e "${YELLOW}Connecting to AWS instance...${NC}"
echo ""

# Connect to the instance and kill the Python training process
SSH_COMMAND="ps aux | grep 'train_branched_regularization_sync_aug_tcn_large.py' | grep -v grep | awk '{print \$2}' | xargs -r kill -9"
ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "$SSH_COMMAND"

# Verify the process has been killed
VERIFY_COMMAND="ps aux | grep 'train_branched_regularization_sync_aug_tcn_large.py' | grep -v grep"
RUNNING_PROCESSES=$(ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "$VERIFY_COMMAND")

if [ -z "$RUNNING_PROCESSES" ]; then
    echo -e "${GREEN}Success: TCN Large training process has been terminated.${NC}"
    echo -e "${YELLOW}You can now deploy the new improved model.${NC}"
else
    echo -e "${RED}Warning: Some training processes may still be running:${NC}"
    echo "$RUNNING_PROCESSES"
    echo ""
    echo -e "${YELLOW}Attempting stronger termination...${NC}"
    
    # Try more aggressive approach - kill any python process
    SSH_COMMAND="ps aux | grep python | grep -v grep | awk '{print \$2}' | xargs -r kill -9"
    ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "$SSH_COMMAND"
    
    echo -e "${GREEN}All training processes should now be terminated.${NC}"
fi

echo -e "${BLUE}=================================================================${NC}"
echo -e "${GREEN}PROCESS TERMINATION COMPLETE${NC}"
echo -e "${BLUE}=================================================================${NC}"
