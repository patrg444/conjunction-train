#!/bin/bash
# Automatic script to clean up the EC2 instance without user intervention
# This script only cleans up files without stopping or terminating the instance

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
echo -e "${BLUE}     AUTO CLEANING EC2 INSTANCE AFTER MODEL DOWNLOAD     ${NC}"
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
    
    # Calculate disk space savings
    echo 'Disk space after cleanup:'
    df -h /home/ec2-user
"
echo -e "${GREEN}Files cleaned up.${NC}"

echo -e "${BLUE}=================================================================${NC}"
echo -e "${GREEN}AUTO CLEANUP PROCESS COMPLETE${NC}"
echo -e "${YELLOW}Note: The instance is still running. To stop or terminate it,${NC}"
echo -e "${YELLOW}please use the interactive cleanup_ec2_instance.sh script.${NC}"
echo -e "${BLUE}=================================================================${NC}"
