#!/bin/bash
# Script to monitor the fixed TCN large model training

# ANSI colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}==========================================================================${NC}"
echo -e "${GREEN}    MONITORING FIXED TCN LARGE MODEL WITH BALANCED HYPERPARAMETERS    ${NC}"
echo -e "${BLUE}==========================================================================${NC}"
echo ""

# Instance details from existing scripts
INSTANCE_IP="3.235.76.0"
USERNAME="ec2-user"
KEY_FILE="./aws-setup/emotion-recognition-key-fixed-20250323090016.pem"
LOG_FILE="training_branched_regularization_sync_aug_tcn_large_fixed.log"

if [ ! -f "$KEY_FILE" ]; then
    echo -e "${RED}Error: SSH key file not found: $KEY_FILE${NC}"
    echo "Please ensure the key file path is correct."
    exit 1
fi

echo -e "${YELLOW}Instance IP:${NC} $INSTANCE_IP"
echo -e "${YELLOW}Log file:${NC} $LOG_FILE"
echo -e "${GREEN}Starting continuous monitoring... (Press Ctrl+C to stop)${NC}"
echo

# Use SSH to continuously monitor the log file
ssh -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "tail -f ~/emotion_training/$LOG_FILE"
