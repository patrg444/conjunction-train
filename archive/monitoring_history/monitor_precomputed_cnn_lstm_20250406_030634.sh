#!/bin/bash
# Monitoring script for Precomputed CNN Audio + LSTM model training # Updated description
# Generated specifically for run started around: 20250406_030634

# ANSI colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

INSTANCE_IP="18.206.48.166" # Updated IP
USERNAME="ec2-user"
KEY_FILE="/Users/patrickgloria/conjunction-train/aws-setup/emotion-recognition-key-fixed-20250323090016.pem" # Use absolute path
REMOTE_DIR="/home/ec2-user/emotion_training"
REMOTE_MONITOR_HELPER="monitor_helper_precomputed_cnn_lstm_20250406_030634.sh" # Name of the helper script on EC2

echo -e "${BLUE}==================================================================${NC}"
echo -e "${GREEN} MONITORING PRECOMPUTED CNN AUDIO + LSTM MODEL TRAINING (20250406_030634) ${NC}" # Updated Title
echo -e "${BLUE}==================================================================${NC}"
echo -e "${YELLOW}Instance:${NC} $USERNAME@$INSTANCE_IP"
echo -e "${YELLOW}Executing remote helper:${NC} $REMOTE_DIR/$REMOTE_MONITOR_HELPER"
echo -e "${BLUE}==================================================================${NC}"

# Start continuous monitoring by cd'ing first, then executing with bash ./script
echo -e "${YELLOW}Starting continuous real-time monitoring... Press Ctrl+C to exit.${NC}"
ssh -t -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "cd /home/ec2-user/emotion_training && bash ./" # cd first, then execute


