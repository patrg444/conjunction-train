#!/bin/bash
# Script to terminate the EC2 instance after model download and cleanup

# ANSI colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Instance details
INSTANCE_IP="98.82.121.48"
KEY_FILE="aws-setup/emotion-recognition-key-20250322082227.pem"

echo -e "${BLUE}=================================================================${NC}"
echo -e "${RED}     TERMINATING EC2 INSTANCE - DISK IS FULL (100%)     ${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "${YELLOW}Instance IP:${NC} $INSTANCE_IP"
echo -e "${BLUE}=================================================================${NC}"
echo ""

# Check if AWS CLI is available
if ! command -v aws &> /dev/null; then
    echo -e "${RED}AWS CLI is not installed. Please install it to proceed with instance termination.${NC}"
    exit 1
fi

# Find the instance ID
echo -e "${YELLOW}Finding instance ID...${NC}"
INSTANCE_ID=$(aws ec2 describe-instances --filters "Name=private-ip-address,Values=$INSTANCE_IP" --query "Reservations[].Instances[].InstanceId" --output text)

if [ -z "$INSTANCE_ID" ]; then
    echo -e "${RED}Could not find instance with IP: $INSTANCE_IP${NC}"
    exit 1
fi

echo -e "${GREEN}Found instance ID: $INSTANCE_ID${NC}"

# Terminate the instance
echo -e "${RED}WARNING: This will permanently delete the instance.${NC}"
echo -e "${YELLOW}Our model has been successfully downloaded and the instance disk is 100% full.${NC}"
echo -e "${YELLOW}Terminating the instance will stop all AWS charges for it.${NC}"
read -p "Are you sure you want to terminate the instance? (y/n): " CONFIRM

if [[ $CONFIRM == [Yy]* ]]; then
    echo -e "${YELLOW}Terminating EC2 instance...${NC}"
    aws ec2 terminate-instances --instance-ids $INSTANCE_ID
    echo -e "${GREEN}Termination request sent. It may take a few minutes to complete.${NC}"
    
    # Check termination status
    echo -e "${YELLOW}Checking termination status...${NC}"
    sleep 10
    STATUS=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query "Reservations[].Instances[].State.Name" --output text)
    echo -e "${GREEN}Instance status: $STATUS${NC}"
    
    if [ "$STATUS" == "shutting-down" ] || [ "$STATUS" == "terminated" ]; then
        echo -e "${GREEN}Instance is being terminated successfully.${NC}"
    else
        echo -e "${YELLOW}Instance status is $STATUS. It may take more time to terminate completely.${NC}"
    fi
else
    echo -e "${YELLOW}Termination cancelled. Instance will continue to incur AWS charges.${NC}"
fi

echo -e "${BLUE}=================================================================${NC}"
echo -e "${GREEN}INSTANCE MANAGEMENT COMPLETE${NC}"
echo -e "${BLUE}=================================================================${NC}"
