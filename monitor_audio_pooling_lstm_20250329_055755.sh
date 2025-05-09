#!/bin/bash
# Monitoring script for Audio Pooling LSTM model training
# Generated specifically for run started around: 20250329_055755

# ANSI colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

INSTANCE_IP="3.235.76.0"
USERNAME="ec2-user"
KEY_FILE="/Users/patrickgloria/conjunction-train/aws-setup/emotion-recognition-key-fixed-20250323090016.pem" # Absolute path
REMOTE_DIR="/home/ec2-user/emotion_training"
REMOTE_MONITOR_HELPER="monitor_helper_audio_pooling_lstm_20250329_055755.sh" # Name of the helper script on EC2

echo -e "${BLUE}==================================================================${NC}"
echo -e "${GREEN}    MONITORING AUDIO POOLING LSTM MODEL TRAINING (20250329_055755)    ${NC}"
echo -e "${BLUE}==================================================================${NC}"
echo -e "${YELLOW}Instance:${NC} $USERNAME@$INSTANCE_IP"
echo -e "${YELLOW}Executing remote helper:${NC} $REMOTE_DIR/$REMOTE_MONITOR_HELPER"
echo -e "${BLUE}==================================================================${NC}"

# Start continuous monitoring by executing the remote helper script
echo -e "${YELLOW}Starting continuous real-time monitoring... Press Ctrl+C to exit.${NC}"
# Use -i with the absolute path directly. Ensure simple quoting.
ssh -t -i "$KEY_FILE" "$USERNAME@$INSTANCE_IP" "cd $REMOTE_DIR && ./$REMOTE_MONITOR_HELPER"
