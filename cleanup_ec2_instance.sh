#!/bin/bash
# Script to clean up the EC2 instance now that we have our model

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
echo -e "${BLUE}     CLEANING UP EC2 INSTANCE AFTER MODEL DOWNLOAD     ${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "${YELLOW}Instance IP:${NC} $INSTANCE_IP"
echo -e "${BLUE}=================================================================${NC}"
echo ""

# Check if instance is available
echo -e "${YELLOW}Checking if the instance is available...${NC}"
if ! ssh -i $KEY_FILE -o StrictHostKeyChecking=no -o ConnectTimeout=5 ec2-user@$INSTANCE_IP echo "SSH connection established" &> /dev/null; then
    echo -e "${RED}Failed to connect to the instance. Please check if it's running.${NC}"
    exit 1
fi
echo -e "${GREEN}Instance is available.${NC}"

# Stop any running training processes
echo -e "${YELLOW}Stopping any running training processes...${NC}"
ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "sudo pkill -f python || true"
echo -e "${GREEN}Training processes stopped.${NC}"

# Remove unnecessary files except for the model we need
echo -e "${YELLOW}Cleaning up unnecessary files...${NC}"
ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "
    echo 'Preserving just the necessary model files...'
    # Make a backup of the model file if it's not already backed up
    if [ -f /home/ec2-user/emotion_training/models/dynamic_padding_no_leakage/model_best.h5 ]; then
        echo 'Backing up model file...'
        mkdir -p /home/ec2-user/model_backup
        cp /home/ec2-user/emotion_training/models/dynamic_padding_no_leakage/model_best.h5 /home/ec2-user/model_backup/
        cp /home/ec2-user/emotion_training/training_no_leakage_conda.log /home/ec2-user/model_backup/
        echo 'Model files backed up to /home/ec2-user/model_backup/'
    fi
    
    # Remove large dataset files to free up space
    echo 'Removing large dataset files...'
    find /home/ec2-user/emotion_training -name '*.npz' -type f -delete
    find /home/ec2-user/emotion_training -name '*.wav' -type f -delete
    find /home/ec2-user/emotion_training -name '*.mp4' -type f -delete
    
    # Clean up temporary files and caches
    echo 'Removing temporary files and caches...'
    find /home/ec2-user/emotion_training -name '__pycache__' -type d -exec rm -rf {} +
    find /home/ec2-user/emotion_training -name '.ipynb_checkpoints' -type d -exec rm -rf {} +
    find /home/ec2-user/emotion_training -name '*.pyc' -type f -delete
    
    # Remove all models except the one we need
    echo 'Removing unnecessary model files...'
    find /home/ec2-user/emotion_training/models -type f -name '*.h5' -not -path '*/dynamic_padding_no_leakage/*' -delete
"
echo -e "${GREEN}Files cleaned up.${NC}"

# Ask if the user wants to stop or terminate the instance
echo -e "${YELLOW}Do you want to stop or terminate the EC2 instance?${NC}"
echo -e "1. Stop (can be restarted later)"
echo -e "2. Terminate (permanently delete the instance)"
echo -e "3. Leave running"
read -p "Enter your choice (1-3): " CHOICE

case $CHOICE in
    1)
        echo -e "${YELLOW}Stopping the EC2 instance...${NC}"
        aws ec2 stop-instances --instance-ids $(aws ec2 describe-instances --filters "Name=private-ip-address,Values=$INSTANCE_IP" --query "Reservations[].Instances[].InstanceId" --output text)
        echo -e "${GREEN}Instance stop request sent. It may take a few minutes to complete.${NC}"
        ;;
    2)
        echo -e "${RED}WARNING: This will permanently delete the instance.${NC}"
        read -p "Are you sure you want to terminate the instance? (y/n): " CONFIRM
        if [[ $CONFIRM == [Yy]* ]]; then
            echo -e "${YELLOW}Terminating the EC2 instance...${NC}"
            aws ec2 terminate-instances --instance-ids $(aws ec2 describe-instances --filters "Name=private-ip-address,Values=$INSTANCE_IP" --query "Reservations[].Instances[].InstanceId" --output text)
            echo -e "${GREEN}Instance termination request sent. It may take a few minutes to complete.${NC}"
        else
            echo -e "${YELLOW}Termination cancelled.${NC}"
        fi
        ;;
    3)
        echo -e "${YELLOW}Leaving the instance running.${NC}"
        echo -e "${RED}Note: Running instances continue to incur costs.${NC}"
        ;;
    *)
        echo -e "${RED}Invalid choice. Leaving the instance running.${NC}"
        echo -e "${RED}Note: Running instances continue to incur costs.${NC}"
        ;;
esac

echo -e "${BLUE}=================================================================${NC}"
echo -e "${GREEN}CLEANUP PROCESS COMPLETE${NC}"
echo -e "${BLUE}=================================================================${NC}"
