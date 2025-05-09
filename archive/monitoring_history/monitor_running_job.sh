#!/bin/bash
# Script to monitor the running emotion recognition training job

# ANSI colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Instance details
INSTANCE_ID="i-0dd2f787db00b205f"
INSTANCE_IP="98.82.121.48"
KEY_FILE="aws-setup/emotion-recognition-key-20250322082227.pem"
LOG_FILE="training_lstm_attention_model.log"  # Assuming standard log name

echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}     MONITORING RUNNING EMOTION RECOGNITION TRAINING JOB         ${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "${YELLOW}Instance ID:${NC} $INSTANCE_ID"
echo -e "${YELLOW}Instance IP:${NC} $INSTANCE_IP"
echo -e "${YELLOW}Instance Type:${NC} c5.24xlarge (CPU-optimized)"
echo -e "${YELLOW}SSH Key:${NC} $KEY_FILE"

# Monitoring options
echo -e ""
echo -e "${BLUE}Monitoring options:${NC}"
echo -e "  1. ${YELLOW}View training logs${NC}"
echo -e "  2. ${YELLOW}Check CPU usage${NC}"
echo -e "  3. ${YELLOW}List training files${NC}" 
echo -e "  4. ${YELLOW}View running processes${NC}"
echo -e ""

read -p "Enter option (1-4): " OPTION

case $OPTION in
  1)
    echo -e "${GREEN}Streaming training log...${NC}"
    ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "find ~/emotion_training -name '*.log' | xargs ls -ltr | tail -n 1"
    echo -e "${YELLOW}Recent logs found above. Streaming latest log...${NC}"
    ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "find ~/emotion_training -name '*.log' | xargs ls -t | head -1 | xargs tail -f"
    ;;
  2)
    echo -e "${GREEN}Checking CPU usage...${NC}"
    ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "top -bn 1 | head -20"
    ;;
  3)
    echo -e "${GREEN}Listing training directory structure...${NC}"
    ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "find ~/emotion_training -maxdepth 2 -type d | sort"
    echo -e "${YELLOW}Listing recently modified files:${NC}"
    ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "find ~/emotion_training -type f -mmin -60 | sort"
    ;;
  4)
    echo -e "${GREEN}Checking running processes...${NC}"
    ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "ps aux | grep python"
    echo -e "${YELLOW}Checking tmux sessions:${NC}"
    ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "tmux list-sessions"
    ;;
  *)
    echo -e "${RED}Invalid option!${NC}"
    ;;
esac
