#!/bin/bash
# Script to monitor CPU usage from running emotion recognition training job

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
echo -e "${BLUE}     MONITORING CPU USAGE FROM EMOTION RECOGNITION TRAINING      ${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "${YELLOW}Instance ID:${NC} $INSTANCE_ID"
echo -e "${YELLOW}Instance IP:${NC} $INSTANCE_IP"
echo -e "${YELLOW}Instance Type:${NC} c5.24xlarge (96 vCPUs)"
echo -e "${YELLOW}Monitoring CPU usage...${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo ""

# Continuous monitoring of CPU usage (refreshes every 2 seconds)
ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "watch -n 2 'top -bn 1 | head -20; echo \"\nPython Processes:\"; ps aux | grep python | grep -v grep'"
