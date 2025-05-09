#!/bin/bash
# Script to stream logs from running emotion recognition training job

# ANSI colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Instance details
INSTANCE_ID="i-0dd2f787db00b205f"
INSTANCE_IP="98.82.121.48"
KEY_FILE="aws-setup/emotion-recognition-key-20250322082227.pem"

echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}     STREAMING LOGS FROM EMOTION RECOGNITION TRAINING JOB        ${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "${YELLOW}Instance ID:${NC} $INSTANCE_ID"
echo -e "${YELLOW}Instance IP:${NC} $INSTANCE_IP"
echo -e "${YELLOW}Streaming training logs...${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo ""

# Find and stream the latest log file
ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "find ~/emotion_training -name '*.log' | xargs ls -ltr | tail -n 3"
echo -e "${YELLOW}Recent logs found above. Streaming latest log...${NC}"
ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "find ~/emotion_training -name '*.log' | xargs ls -t | head -1 | xargs tail -f"
