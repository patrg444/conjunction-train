#!/bin/bash
# Script to check training files and identify running model

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
echo -e "${BLUE}     CHECKING TRAINING FILES AND MODEL CONFIGURATION             ${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "${YELLOW}Instance ID:${NC} $INSTANCE_ID"
echo -e "${YELLOW}Instance IP:${NC} $INSTANCE_IP"
echo -e "${BLUE}=================================================================${NC}"
echo ""

# Check training directory structure
echo -e "${YELLOW}Directory Structure:${NC}"
ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "find ~/emotion_training -maxdepth 2 -type d | sort"
echo ""

# Check training scripts
echo -e "${YELLOW}Available Training Scripts:${NC}"
ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "find ~/emotion_training/scripts -name 'train*.py' | sort"
echo ""

# Check running processes to identify which script is active
echo -e "${YELLOW}Running Python Processes:${NC}"
ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "ps aux | grep python | grep -v grep"
echo ""

# Check tmux sessions
echo -e "${YELLOW}Active Tmux Sessions:${NC}"
ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "tmux list-sessions 2>/dev/null || echo 'No tmux sessions found'"
echo ""

# Check most recently modified model files
echo -e "${YELLOW}Most Recently Modified Model Files:${NC}"
ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "find ~/emotion_training/models -type f -mmin -60 2>/dev/null | sort || echo 'No recently modified model files found'"
echo ""

# Check log summary (last 10 lines)
echo -e "${YELLOW}Latest Log Summary:${NC}"
ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "find ~/emotion_training -name '*.log' | xargs ls -t 2>/dev/null | head -1 | xargs tail -10 2>/dev/null || echo 'No log files found'"
