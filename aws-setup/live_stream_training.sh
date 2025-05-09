#!/bin/bash
# Live streaming of LSTM attention model training logs
# This script establishes a continuous connection to the EC2 instance and streams the logs in real-time

# ANSI color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Source connection details
if [ ! -f "aws-setup/lstm_attention_model_connection.txt" ]; then
    echo -e "${RED}Error: Connection details file not found.${NC}"
    echo "Please ensure aws-setup/lstm_attention_model_connection.txt exists."
    exit 1
fi

source aws-setup/lstm_attention_model_connection.txt

echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}     LIVE STREAMING LSTM ATTENTION MODEL TRAINING LOGS          ${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "${YELLOW}Instance ID:${NC} $INSTANCE_ID"
echo -e "${YELLOW}Instance IP:${NC} $INSTANCE_IP"
echo -e "${YELLOW}Started at:${NC} $(date)"
echo -e "${BLUE}=================================================================${NC}"

# Check instance status
echo -e "${YELLOW}Checking instance status...${NC}"
INSTANCE_STATE=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].State.Name' --output text)

if [ "$INSTANCE_STATE" != "running" ]; then
    echo -e "${RED}Error: Instance is not running (current state: $INSTANCE_STATE).${NC}"
    exit 1
fi

echo -e "${GREEN}Instance status: $INSTANCE_STATE${NC}"

# Establish continuous SSH connection and stream logs in real-time
echo -e "${YELLOW}Establishing continuous connection to training logs...${NC}"
echo -e "${BLUE}Press Ctrl+C to exit the stream${NC}"
echo -e "${BLUE}=================================================================${NC}"

# Connect to the instance and stream the logs
ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "tail -f ~/emotion_training/training_lstm_attention_no_aug.log"
