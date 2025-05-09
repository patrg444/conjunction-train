#!/bin/bash
# Deploy and train the enhanced emotion recognition model with:
# 1. Temporal Attention Mechanism
# 2. Focal Loss
# 3. Audio Data Augmentation

# Set script to exit on error
set -e

# ANSI colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Define variables
INSTANCE_TYPE="c5.4xlarge"  # Powerful CPU instance for faster training
INSTANCE_NAME="emotion-recognition-attention-model"
KEY_NAME="emotion-recognition-key-$(date +%Y%m%d%H%M%S)"
REGION="us-east-1"
AMI_ID="ami-0261755bbcb8c4a84"  # Amazon Linux 2
SECURITY_GROUP="emotion-recognition-sg"
LOG_FILE="training_attention_model.log"

# Print banner
echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}    DEPLOYING ENHANCED EMOTION RECOGNITION MODEL TO AWS${NC}"
echo -e "${BLUE}    - With Temporal Attention${NC}"
echo -e "${BLUE}    - With Focal Loss${NC}"
echo -e "${BLUE}    - With Audio Data Augmentation${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo ""

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}AWS CLI is not installed. Please install it first.${NC}"
    exit 1
fi

# Create a tarball of the project without large unnecessary files
echo -e "${YELLOW}Creating project archive...${NC}"
mkdir -p tmp
tar --exclude="__pycache__" \
    --exclude="*.tar.gz" \
    --exclude="*.mp4" \
    --exclude="*.avi" \
    --exclude="*.mp3" \
    --exclude="*.wav" \
    --exclude=".git" \
    --exclude="downsampled_videos" \
    -czf tmp/project.tar.gz .

echo -e "${GREEN}Project archive created.${NC}"

# Create a new key pair for SSH access
echo -e "${YELLOW}Creating SSH key pair...${NC}"
aws ec2 create-key-pair --region $REGION --key-name $KEY_NAME --query 'KeyMaterial' --output text > aws-setup/$KEY_NAME.pem
chmod 400 aws-setup/$KEY_NAME.pem
echo -e "${GREEN}SSH key pair created: ${KEY_NAME}${NC}"

# Create security group if it doesn't exist
if ! aws ec2 describe-security-groups --region $REGION --group-names $SECURITY_GROUP &> /dev/null; then
    echo -e "${YELLOW}Creating security group...${NC}"
    SECURITY_GROUP_ID=$(aws ec2 create-security-group --region $REGION --group-name $SECURITY_GROUP --description "Security group for emotion recognition training" --query 'GroupId' --output text)
    
    # Allow SSH access
    aws ec2 authorize-security-group-ingress --region $REGION --group-id $SECURITY_GROUP_ID --protocol tcp --port 22 --cidr 0.0.0.0/0
    
    echo -e "${GREEN}Security group created: ${SECURITY_GROUP}${NC}"
else
    SECURITY_GROUP_ID=$(aws ec2 describe-security-groups --region $REGION --group-names $SECURITY_GROUP --query 'SecurityGroups[0].GroupId' --output text)
    echo -e "${GREEN}Using existing security group: ${SECURITY_GROUP}${NC}"
fi

# Launch EC2 instance
echo -e "${YELLOW}Launching EC2 instance...${NC}"
INSTANCE_ID=$(aws ec2 run-instances \
    --region $REGION \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-group-ids $SECURITY_GROUP_ID \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME}]" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo -e "${GREEN}EC2 instance launched: ${INSTANCE_ID}${NC}"
echo -e "${YELLOW}Waiting for instance to initialize...${NC}"

# Wait for instance to be running
aws ec2 wait instance-running --region $REGION --instance-ids $INSTANCE_ID
echo -e "${GREEN}Instance is running.${NC}"

# Get public IP address
INSTANCE_IP=$(aws ec2 describe-instances \
    --region $REGION \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo -e "${GREEN}Instance public IP: ${INSTANCE_IP}${NC}"
echo -e "${YELLOW}Waiting for SSH to be available...${NC}"

# Wait for SSH to be available (may take a minute or two)
while ! ssh -i aws-setup/$KEY_NAME.pem -o StrictHostKeyChecking=no -o ConnectTimeout=5 ec2-user@$INSTANCE_IP echo "SSH connection established" &> /dev/null; do
    echo -e "${YELLOW}Still waiting for SSH connection...${NC}"
    sleep 10
done

echo -e "${GREEN}SSH connection established.${NC}"

# Copy project files to instance
echo -e "${YELLOW}Copying project files to instance...${NC}"
scp -i aws-setup/$KEY_NAME.pem -o StrictHostKeyChecking=no tmp/project.tar.gz ec2-user@$INSTANCE_IP:~/project.tar.gz

echo -e "${GREEN}Project files copied.${NC}"

# Set up the instance and start training
echo -e "${YELLOW}Setting up the environment and starting training...${NC}"
ssh -i aws-setup/$KEY_NAME.pem -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP << EOF
    # Create directories
    mkdir -p ~/emotion_training

    # Extract project
    tar -xzf project.tar.gz -C ~/emotion_training

    # Install dependencies
    sudo yum update -y
    sudo yum install -y python3 python3-pip git tmux
    
    # Install Python packages
    cd ~/emotion_training
    pip3 install --user -r requirements.txt
    pip3 install --user tensorflow scipy matplotlib

    # Start training in a tmux session so it continues running after disconnection
    tmux new-session -d -s training "cd ~/emotion_training && python3 scripts/train_branched_attention.py > $LOG_FILE 2>&1"
    
    echo "Training started in tmux session. Use 'tmux attach -t training' to view."
    echo "Log file: ~/emotion_training/$LOG_FILE"
EOF

echo -e "${GREEN}Setup complete and training started.${NC}"
echo ""
echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}                      DEPLOYMENT SUMMARY${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "${YELLOW}Instance ID:${NC} $INSTANCE_ID"
echo -e "${YELLOW}Instance IP:${NC} $INSTANCE_IP"
echo -e "${YELLOW}SSH Key:${NC} aws-setup/$KEY_NAME.pem"
echo ""
echo -e "${YELLOW}To connect to the instance:${NC}"
echo -e "ssh -i aws-setup/$KEY_NAME.pem ec2-user@$INSTANCE_IP"
echo ""
echo -e "${YELLOW}To monitor training:${NC}"
echo -e "ssh -i aws-setup/$KEY_NAME.pem ec2-user@$INSTANCE_IP 'tail -f ~/emotion_training/$LOG_FILE'"
echo ""
echo -e "${YELLOW}To download results when training completes:${NC}"
echo -e "scp -i aws-setup/$KEY_NAME.pem -r ec2-user@$INSTANCE_IP:~/emotion_training/models/attention_focal_loss ."
echo -e "${BLUE}=================================================================${NC}"

# Save connection details for future use
cat > aws-setup/attention_model_connection.txt << EOL
INSTANCE_ID=$INSTANCE_ID
INSTANCE_IP=$INSTANCE_IP
KEY_NAME=$KEY_NAME
KEY_FILE=aws-setup/$KEY_NAME.pem
LOG_FILE=$LOG_FILE
EOL

echo -e "${GREEN}Connection details saved to aws-setup/attention_model_connection.txt${NC}"

# Create a monitoring script
cat > aws-setup/monitor_attention_model.sh << EOL
#!/bin/bash
# Script to monitor training progress of the attention model

# Source connection details
source aws-setup/attention_model_connection.txt

# ANSI colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "\${BLUE}=================================================================\${NC}"
echo -e "\${BLUE}     MONITORING ENHANCED MODEL TRAINING (ATTENTION, FOCAL LOSS)  \${NC}"
echo -e "\${BLUE}=================================================================\${NC}"
echo -e "\${YELLOW}Instance:\${NC} \$INSTANCE_IP"
echo -e "\${YELLOW}Log file:\${NC} ~/emotion_training/\$LOG_FILE"
echo -e "\${GREEN}Streaming training log...\${NC}"
echo -e "\${BLUE}=================================================================\${NC}"
echo ""

ssh -i \$KEY_FILE -o StrictHostKeyChecking=no ec2-user@\$INSTANCE_IP "tail -f ~/emotion_training/\$LOG_FILE"
EOL

chmod +x aws-setup/monitor_attention_model.sh

echo -e "${GREEN}Created monitoring script: aws-setup/monitor_attention_model.sh${NC}"
echo -e "${YELLOW}Run this script to monitor the training progress.${NC}"
