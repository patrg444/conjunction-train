#!/bin/bash
# Script to increase the storage space of an EC2 instance by 5GB

# ANSI colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Instance details
INSTANCE_IP="98.82.121.48"
KEY_FILE="aws-setup/emotion-recognition-key-20250322082227.pem"
ADDITIONAL_GB=5  # Amount of space to add in GB

echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}     INCREASING EC2 INSTANCE STORAGE BY ${ADDITIONAL_GB}GB     ${NC}"
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

# Get instance ID directly from the instance metadata
echo -e "${YELLOW}Getting instance ID directly from the instance...${NC}"
INSTANCE_ID=$(ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "curl -s http://169.254.169.254/latest/meta-data/instance-id")
if [ -z "$INSTANCE_ID" ]; then
    echo -e "${RED}Could not determine instance ID.${NC}"
    exit 1
fi
echo -e "${GREEN}Found instance ID: $INSTANCE_ID${NC}"

# Get volume ID
echo -e "${YELLOW}Finding volume ID...${NC}"
VOLUME_ID=$(aws ec2 describe-volumes --filters "Name=attachment.instance-id,Values=$INSTANCE_ID" --query "Volumes[0].VolumeId" --output text)
if [ -z "$VOLUME_ID" ]; then
    echo -e "${RED}Could not find volume ID for instance: $INSTANCE_ID${NC}"
    exit 1
fi
echo -e "${GREEN}Found volume ID: $VOLUME_ID${NC}"

# Get current volume size
echo -e "${YELLOW}Getting current volume size...${NC}"
CURRENT_SIZE=$(aws ec2 describe-volumes --volume-ids $VOLUME_ID --query "Volumes[0].Size" --output text)
if [ -z "$CURRENT_SIZE" ]; then
    echo -e "${RED}Could not determine current volume size.${NC}"
    exit 1
fi
echo -e "${GREEN}Current volume size: ${CURRENT_SIZE}GB${NC}"

# Calculate new size
NEW_SIZE=$((CURRENT_SIZE + ADDITIONAL_GB))
echo -e "${YELLOW}New volume size will be: ${NEW_SIZE}GB${NC}"

# Auto confirm
echo -e "${YELLOW}This will modify the EBS volume and increase its size to add ${ADDITIONAL_GB}GB. Additional charges may apply.${NC}"
echo -e "${GREEN}Auto-confirming operation...${NC}"
CONFIRM="y"

# Modify volume
echo -e "${YELLOW}Modifying volume...${NC}"
aws ec2 modify-volume --volume-id $VOLUME_ID --size $NEW_SIZE

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to modify volume.${NC}"
    exit 1
fi
echo -e "${GREEN}Volume modification initiated.${NC}"

# Wait for modification to complete
echo -e "${YELLOW}Waiting for volume modification to complete. This may take a few minutes...${NC}"
aws ec2 wait volume-available --volume-ids $VOLUME_ID
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Wait command failed, but modification may still be in progress.${NC}"
fi

# Check volume modification status
VOL_MOD_STATUS=$(aws ec2 describe-volumes-modifications --volume-ids $VOLUME_ID --query "VolumesModifications[0].ModificationState" --output text)
echo -e "${GREEN}Volume modification status: $VOL_MOD_STATUS${NC}"

# Resize the filesystem on the instance
echo -e "${YELLOW}Resizing filesystem on the instance...${NC}"
ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "
    echo 'Checking for new disk space...'
    lsblk
    
    # Resize the file system
    echo 'Resizing the filesystem...'
    sudo growpart /dev/nvme0n1 1 || true
    sudo xfs_growfs -d / || sudo resize2fs /dev/nvme0n1p1 || true
    
    # Show new disk space
    echo 'New disk space:'
    df -h
"

echo -e "${BLUE}=================================================================${NC}"
echo -e "${GREEN}STORAGE INCREASE OPERATION COMPLETE${NC}"
echo -e "${YELLOW}Note: It may take a few minutes for the full space to be available.${NC}"
echo -e "${BLUE}=================================================================${NC}"
