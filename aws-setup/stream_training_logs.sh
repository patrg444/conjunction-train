#!/bin/bash
# Script to continuously stream training logs from the AWS instance to terminal
# Press Ctrl+C to exit

# Set up variables
INSTANCE_IP="98.82.121.48"
USERNAME="ec2-user"
KEY_FILE="emotion-recognition-key-20250322082227.pem"
REMOTE_DIR="~/emotion_training"
LOG_FILE="training_no_leakage_conda.log"

# ANSI color codes for better readability
BLUE='\033[1;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}==================================================================${NC}"
echo -e "${BLUE}       LIVE TRAINING LOG STREAM - NO DATA LEAKAGE (CONDA)${NC}"
echo -e "${BLUE}==================================================================${NC}"
echo -e "${YELLOW}Server:${NC} ${INSTANCE_IP}"
echo -e "${YELLOW}Log file:${NC} ${REMOTE_DIR}/${LOG_FILE}"
echo -e "${GREEN}Streaming log updates in real-time. Press Ctrl+C to exit.${NC}"
echo -e "${BLUE}==================================================================${NC}"
echo ""

# Check if the training process is running
echo -e "${YELLOW}Checking if training process is active...${NC}"
PROCESS_RUNNING=$(ssh -i "${KEY_FILE}" -o StrictHostKeyChecking=no ${USERNAME}@${INSTANCE_IP} "ps aux | grep train_branched_no_leakage.py | grep -v grep | wc -l")

if [ "$PROCESS_RUNNING" -eq "0" ]; then
    echo -e "${RED}Warning: Training process does not appear to be running!${NC}"
    echo -e "${YELLOW}Will still attempt to stream log file in case it starts later.${NC}"
else
    echo -e "${GREEN}Training process is active.${NC}"
    # Get the PID of the training process
    PID=$(ssh -i "${KEY_FILE}" -o StrictHostKeyChecking=no ${USERNAME}@${INSTANCE_IP} "ps aux | grep train_branched_no_leakage.py | grep -v grep | awk '{print \$2}'")
    echo -e "${YELLOW}Process ID:${NC} ${PID}"
fi

echo ""
echo -e "${BLUE}Starting live log stream...${NC}"
echo -e "${BLUE}==================================================================${NC}"
echo ""

# Stream the log file in real-time (tail -f)
ssh -i "${KEY_FILE}" -o StrictHostKeyChecking=no ${USERNAME}@${INSTANCE_IP} "tail -f ${REMOTE_DIR}/${LOG_FILE}"
