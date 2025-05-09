#!/bin/bash
# AWS CLI Setup Script for Emotion Recognition Training
# This script sets up AWS CLI, launches an EC2 instance, and prepares it for training

# Set up colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}==========================================================================${NC}"
echo -e "${BLUE}                AUTOMATED AWS INSTANCE SETUP                              ${NC}"
echo -e "${BLUE}==========================================================================${NC}"
echo ""

# Check for AWS CLI
if ! command -v aws &> /dev/null; then
    echo -e "${RED}Error: AWS CLI is not installed. Please install it first.${NC}"
    echo "Visit: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
    exit 1
fi

# Configure AWS CLI
echo -e "${YELLOW}Sourcing AWS credentials...${NC}"
# Source the credentials file
if [ -f "aws-setup/aws_credentials.sh" ]; then
    source "aws-setup/aws_credentials.sh"
    echo -e "${GREEN}AWS credentials sourced.${NC}"
else
    echo -e "${RED}Error: aws-setup/aws_credentials.sh not found. Please create it with your AWS credentials.${NC}"
    exit 1
fi

# AWS CLI will use environment variables AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_DEFAULT_REGION
# No need to write to ~/.aws/credentials or ~/.aws/config directly if these are set.
# However, creating a minimal config for region and output can still be useful.
mkdir -p ~/.aws
cat > ~/.aws/config << EOF
[default]
region = \${AWS_DEFAULT_REGION:-us-east-1}
output = json
EOF
echo -e "${GREEN}AWS CLI config file updated (region/output). Credentials will be used from environment variables.${NC}"
echo ""

# Select instance type
echo -e "${YELLOW}Select EC2 instance type:${NC}"
echo "1) g4dn.xlarge - Good balance of performance and cost (~$0.52/hour)"
echo "2) r5.2xlarge - High RAM instance (~$0.50/hour)"
echo "3) p3.8xlarge - Maximum performance with 4 V100 GPUs (~$12.24/hour)"
read -p "Enter choice [1-3]: " instance_choice

case $instance_choice in
    1) INSTANCE_TYPE="g4dn.xlarge";;
    2) INSTANCE_TYPE="r5.2xlarge";;
    3) INSTANCE_TYPE="p3.8xlarge";;
    *) echo -e "${RED}Invalid choice, defaulting to g4dn.xlarge${NC}"; INSTANCE_TYPE="g4dn.xlarge";;
esac
echo -e "${GREEN}Selected instance type: $INSTANCE_TYPE${NC}"
echo ""

# Create key pair
KEY_NAME="emotion-recognition-key-$(date +%Y%m%d%H%M%S)"
echo -e "${YELLOW}Creating key pair: $KEY_NAME${NC}"
aws ec2 create-key-pair --key-name "$KEY_NAME" --query 'KeyMaterial' --output text > "$KEY_NAME.pem"
chmod 400 "$KEY_NAME.pem"
echo -e "${GREEN}Key pair created and saved to $KEY_NAME.pem${NC}"
echo ""

# Get Deep Learning AMI ID
echo -e "${YELLOW}Finding latest Deep Learning AMI...${NC}"
# This command looks for the most recent AWS Deep Learning AMI with CUDA support
AMI_ID=$(aws ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=*Deep Learning AMI*" "Name=description,Values=*CUDA*" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text)

if [ -z "$AMI_ID" ]; then
    echo -e "${RED}Could not find Deep Learning AMI, using Amazon Linux 2 instead.${NC}"
    # Fallback to Amazon Linux 2
    AMI_ID=$(aws ec2 describe-images \
        --owners amazon \
        --filters "Name=name,Values=amzn2-ami-hvm-*-x86_64-gp2" \
        --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
        --output text)
fi

echo -e "${GREEN}Using AMI: $AMI_ID${NC}"
echo ""

# Create security group
SECURITY_GROUP_NAME="emotion-recognition-sg-$(date +%Y%m%d%H%M%S)"
echo -e "${YELLOW}Creating security group: $SECURITY_GROUP_NAME${NC}"
SECURITY_GROUP_ID=$(aws ec2 create-security-group \
    --group-name "$SECURITY_GROUP_NAME" \
    --description "Security group for emotion recognition training" \
    --query 'GroupId' \
    --output text)

# Allow SSH access
aws ec2 authorize-security-group-ingress \
    --group-id "$SECURITY_GROUP_ID" \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0

echo -e "${GREEN}Security group created and configured: $SECURITY_GROUP_ID${NC}"
echo ""

# Launch EC2 instance
echo -e "${YELLOW}Launching EC2 instance of type $INSTANCE_TYPE...${NC}"
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SECURITY_GROUP_ID" \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":100,\"VolumeType\":\"gp2\"}}]" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=EmotionRecognitionTraining}]" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo -e "${BLUE}Instance ID: $INSTANCE_ID${NC}"

# Wait for instance to be running
echo -e "${YELLOW}Waiting for instance to start...${NC}"
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"

# Get instance public IP
INSTANCE_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo -e "${GREEN}Instance is running at IP: $INSTANCE_IP${NC}"
echo ""

# Wait a bit more for the instance to initialize
echo -e "${YELLOW}Waiting for instance to complete initialization (60 seconds)...${NC}"
sleep 60

# Create setup script for the instance
echo -e "${YELLOW}Preparing setup scripts...${NC}"
cat > setup_instance.sh << EOF
#!/bin/bash
# This script will run on the EC2 instance to set up the environment

# Determine the correct user account
if id -u ec2-user >/dev/null 2>&1; then
  USER_ACCOUNT="ec2-user"
elif id -u ubuntu >/dev/null 2>&1; then
  USER_ACCOUNT="ubuntu"
else
  USER_ACCOUNT=\$(whoami)
fi

# Upload our setup package
echo "Uploading training package..."
scp -i "$KEY_NAME.pem" -o StrictHostKeyChecking=no aws-emotion-recognition-training.tar.gz \$USER_ACCOUNT@$INSTANCE_IP:~/

# SSH into the instance and run setup commands
echo "Setting up the instance environment..."
ssh -i "$KEY_NAME.pem" -o StrictHostKeyChecking=no \$USER_ACCOUNT@$INSTANCE_IP << 'ENDSSH'
# Create directories
mkdir -p ~/emotion_training
tar -xzf aws-emotion-recognition-training.tar.gz -C ~/emotion_training
cd ~/emotion_training

# Make scripts executable
chmod +x aws-instance-setup.sh train-on-aws.sh

# Run setup script
echo "Running environment setup script..."
./aws-instance-setup.sh

# Start training with default parameters
echo "Starting training..."
nohup ./train-on-aws.sh > training.log 2>&1 &
echo "Training started in background. You can check progress with: tail -f training.log"
ENDSSH

# Create a script to check training progress
cat > check_progress.sh << EOF2
#!/bin/bash
ssh -i "$KEY_NAME.pem" -o StrictHostKeyChecking=no \$USER_ACCOUNT@$INSTANCE_IP "tail -f ~/emotion_training/training.log"
EOF2
chmod +x check_progress.sh

# Create a script to download results
cat > download_results.sh << EOF3
#!/bin/bash
mkdir -p results
scp -i "$KEY_NAME.pem" -r \$USER_ACCOUNT@$INSTANCE_IP:~/emotion_training/models/branched_6class results/
echo "Results downloaded to results/branched_6class"
EOF3
chmod +x download_results.sh

# Create a script to stop/terminate the instance
cat > stop_instance.sh << EOF4
#!/bin/bash
# This script stops or terminates the EC2 instance

echo "Do you want to stop or terminate the instance?"
echo "1) Stop instance (can be restarted later, storage charges apply)"
echo "2) Terminate instance (permanent deletion, no further charges)"
read -p "Enter choice [1-2]: " choice

case \$choice in
    1) 
        echo "Stopping instance $INSTANCE_ID..."
        aws ec2 stop-instances --instance-ids "$INSTANCE_ID"
        echo "Instance stopped. To restart it later, use: aws ec2 start-instances --instance-ids $INSTANCE_ID"
        ;;
    2) 
        echo "Terminating instance $INSTANCE_ID..."
        aws ec2 terminate-instances --instance-ids "$INSTANCE_ID"
        echo "Instance terminated."
        ;;
    *) 
        echo "Invalid choice, no action taken." 
        ;;
esac
EOF4
chmod +x stop_instance.sh
EOF
chmod +x setup_instance.sh

# Execute the setup script
echo -e "${YELLOW}Executing setup script...${NC}"
./setup_instance.sh

echo -e "${GREEN}Setup complete!${NC}"
echo -e "${BLUE}Instance ID: $INSTANCE_ID${NC}"
echo -e "${BLUE}Instance IP: $INSTANCE_IP${NC}"
echo -e "${BLUE}SSH Key: $KEY_NAME.pem${NC}"
echo ""
echo -e "${YELLOW}Available commands:${NC}"
echo "- ./check_progress.sh - Check training progress"
echo "- ./download_results.sh - Download trained models"
echo "- ./stop_instance.sh - Stop or terminate the instance when done"
echo ""
echo -e "${RED}IMPORTANT: Remember to stop or terminate your instance when done to avoid unnecessary charges.${NC}"
