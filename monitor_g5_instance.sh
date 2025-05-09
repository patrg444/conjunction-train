#!/bin/bash
# Script to monitor G5 instance until it's ready and then connect

# Configuration - edit these values as needed
INSTANCE_ID="i-06aa5ac2a37a6fbb3"
SSH_KEY_PATH="/Users/patrickgloria/Downloads/gpu-key.pem"
REGION="us-east-1"  # Default AWS region

# Color formatting
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Monitoring instance $INSTANCE_ID until ready...${NC}"
echo -e "Press Ctrl+C to cancel at any time\n"

while true; do
  # Get instance details
  INSTANCE_JSON=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --region $REGION 2>/dev/null)
  
  if [ $? -ne 0 ]; then
    echo -e "${RED}Error getting instance details. Check your AWS CLI configuration and permissions.${NC}"
    exit 1
  fi
  
  # Extract state and IP information
  STATE=$(echo $INSTANCE_JSON | jq -r '.Reservations[0].Instances[0].State.Name')
  PUBLIC_IP=$(echo $INSTANCE_JSON | jq -r '.Reservations[0].Instances[0].PublicIpAddress')
  INSTANCE_TYPE=$(echo $INSTANCE_JSON | jq -r '.Reservations[0].Instances[0].InstanceType')
  
  # Display current status
  echo -e "Time: $(date)"
  echo -e "Instance: ${YELLOW}$INSTANCE_ID${NC}"
  echo -e "Type: ${YELLOW}$INSTANCE_TYPE${NC}"
  echo -e "State: ${YELLOW}$STATE${NC}"
  echo -e "Public IP: ${YELLOW}${PUBLIC_IP:-None assigned yet}${NC}\n"
  
  # Check if running and has IP
  if [[ "$STATE" == "running" && "$PUBLIC_IP" != "null" ]]; then
    echo -e "${GREEN}Instance is ready! Connecting via SSH...${NC}"
    echo -e "SSH command: ssh -i $SSH_KEY_PATH ubuntu@$PUBLIC_IP\n"
    
    # Check if SSH port is open (with timeout)
    echo "Verifying SSH port is open..."
    timeout 5 bash -c "</dev/tcp/$PUBLIC_IP/22" 2>/dev/null
    if [ $? -eq 0 ]; then
      echo -e "${GREEN}SSH port is open. Connecting...${NC}\n"
      ssh -i $SSH_KEY_PATH ubuntu@$PUBLIC_IP
      exit 0
    else
      echo -e "${YELLOW}SSH port not responding yet. Will retry in 15 seconds...${NC}\n"
    fi
  else
    echo -e "Instance not ready yet. Will check again in 15 seconds...\n"
  fi
  
  sleep 15
done
