#!/bin/bash
# Script to monitor training progress of the LSTM attention model

# Source connection details
source aws-setup/lstm_attention_model_connection.txt

# ANSI colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}     MONITORING LSTM ATTENTION MODEL TRAINING                    ${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "${YELLOW}Instance:${NC} $INSTANCE_IP"
echo -e "${YELLOW}Log file:${NC} ~/emotion_training/$LOG_FILE"
echo -e "${GREEN}Streaming training log...${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo ""

# Option for CPU monitoring
if [ "$1" == "--cpu" ]; then
    echo -e "${YELLOW}Monitoring CPU usage...${NC}"
    ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "while true; do clear; top -b -n 1 | head -20; sleep 5; done"
    exit 0
fi

# Stream the log
ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "tail -f ~/emotion_training/$LOG_FILE"
