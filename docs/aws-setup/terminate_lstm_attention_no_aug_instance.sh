#!/bin/bash
# Script to terminate the LSTM attention (no augmentation) EC2 instance

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
echo -e "${BLUE}     TERMINATING LSTM ATTENTION (NO AUG) EC2 INSTANCE           ${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "${YELLOW}Instance ID:${NC} $INSTANCE_ID"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}Error: AWS CLI is not installed. Please install it first.${NC}"
    exit 1
fi

# Confirm with user
echo -e "${RED}WARNING: This will terminate the EC2 instance permanently.${NC}"
echo -e "${RED}All data not saved locally will be lost.${NC}"
read -p "Are you sure you want to continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Operation cancelled.${NC}"
    exit 0
fi

# Check if instance exists
echo -e "${YELLOW}Checking instance status...${NC}"
INSTANCE_STATUS=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null)

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Could not find instance $INSTANCE_ID. It may already be terminated.${NC}"
    exit 1
fi

echo -e "${GREEN}Instance found with status: $INSTANCE_STATUS${NC}"

# Terminate the instance
echo -e "${YELLOW}Terminating instance...${NC}"
aws ec2 terminate-instances --instance-ids $INSTANCE_ID

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to terminate instance.${NC}"
    exit 1
fi

echo -e "${GREEN}Termination request sent successfully.${NC}"

# Wait for termination to complete
echo -e "${YELLOW}Waiting for instance to terminate. This may take a minute...${NC}"
aws ec2 wait instance-terminated --instance-ids $INSTANCE_ID

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Warning: Wait command failed, but termination may still be in progress.${NC}"
    echo -e "${YELLOW}Please check the AWS console to confirm termination.${NC}"
    exit 1
fi

echo -e "${GREEN}Instance successfully terminated.${NC}"

# Clean up key pair
echo -e "${YELLOW}Deleting key pair...${NC}"
KEY_NAME="emotion-recognition-key-lstm-attention-no-aug"
aws ec2 delete-key-pair --key-name $KEY_NAME

echo -e "${GREEN}===============================================================${NC}"
echo -e "${GREEN}EC2 instance terminated and cleaned up successfully!${NC}"
echo -e "${GREEN}===============================================================${NC}"
