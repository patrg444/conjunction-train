#!/bin/bash
# Script to launch a c5.24xlarge instance for emotion recognition training

# Disable the AWS CLI pager to avoid interactive prompts
export AWS_PAGER=""

# Set up colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}==========================================================================${NC}"
echo -e "${BLUE}              HIGH-PERFORMANCE c5.24xlarge INSTANCE SETUP                 ${NC}"
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
elif [ -f "./aws_credentials.sh" ]; then # If script is run from within aws-setup/
    source "./aws_credentials.sh"
    echo -e "${GREEN}AWS credentials sourced.${NC}"
else
    echo -e "${RED}Error: aws_credentials.sh not found in aws-setup/ or current directory. Please create it with your AWS credentials.${NC}"
    exit 1
fi

# AWS CLI will use environment variables AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_DEFAULT_REGION
# No need to write to ~/.aws/credentials directly if these are set.
# However, creating a minimal config for region and output can still be useful.
mkdir -p ~/.aws
cat > ~/.aws/config << 'CONF'
[default]
region = \${AWS_DEFAULT_REGION:-us-east-1}
output = json
CONF
echo -e "${GREEN}AWS CLI config file updated (region/output). Credentials will be used from environment variables.${NC}"
echo ""

# Define the instance type (no fallback in this script)
INSTANCE_TYPE="c5.24xlarge"  # 96 vCPUs, high compute

echo -e "${YELLOW}Will attempt to launch ${BLUE}$INSTANCE_TYPE${YELLOW} instance${NC}"
echo ""

# Create key pair
KEY_NAME="emotion-recognition-key-$(date +%Y%m%d%H%M%S)"
echo -e "${YELLOW}Creating key pair: $KEY_NAME${NC}"
aws ec2 create-key-pair --key-name "$KEY_NAME" --query 'KeyMaterial' --output text > "$KEY_NAME.pem"
chmod 400 "$KEY_NAME.pem"
echo -e "${GREEN}Key pair created and saved to $KEY_NAME.pem${NC}"
echo ""

# Find Amazon Deep Learning AMI
echo -e "${YELLOW}Finding Amazon Deep Learning AMI...${NC}"
# Look specifically for the Deep Learning AMI (Amazon Linux 2) Version
AMI_ID=$(aws ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=*Deep Learning AMI (Amazon Linux 2)*" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text)

if [ -z "$AMI_ID" ] || [ "$AMI_ID" = "None" ]; then
    echo -e "${YELLOW}Trying alternative AMI search...${NC}"
    # Try another approach to find the Deep Learning AMI
    AMI_ID=$(aws ec2 describe-images \
        --owners amazon \
        --filters "Name=name,Values=*Deep Learning AMI*" \
        --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
        --output text)
fi

if [ -z "$AMI_ID" ] || [ "$AMI_ID" = "None" ]; then
    echo -e "${RED}Could not find Deep Learning AMI. Using Amazon Linux 2 instead.${NC}"
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

# Try to launch the c5.24xlarge instance
echo -e "${YELLOW}Attempting to launch instance of type ${BLUE}$INSTANCE_TYPE${YELLOW}...${NC}"

# Try to launch this instance type
LAUNCH_OUTPUT=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SECURITY_GROUP_ID" \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":100,\"VolumeType\":\"gp2\"}}]" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=EmotionRecognitionTraining}]" \
    --query 'Instances[0].InstanceId' \
    --output text 2>&1)

# Check if the launch was successful
if [[ $LAUNCH_OUTPUT == i-* ]]; then
    INSTANCE_ID=$LAUNCH_OUTPUT
    echo -e "${GREEN}Successfully launched $INSTANCE_TYPE instance with ID: $INSTANCE_ID${NC}"
else
    echo -e "${RED}Failed to launch $INSTANCE_TYPE: $LAUNCH_OUTPUT${NC}"
    echo -e "${RED}Please try the aws_cpu_fallback.sh script which will attempt other instance types.${NC}"
    exit 1
fi

echo -e "${BLUE}Instance ID: $INSTANCE_ID${NC}"
echo -e "${BLUE}Instance Type: $INSTANCE_TYPE${NC}"

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
cat > setup_instance.sh << SETUPEOF
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

# Create an optimized setup script for the instance based on the instance type
cat > cpu_setup.sh << 'CPUEOF'
#!/bin/bash
# CPU-specific optimizations for c5.24xlarge

# Configure TensorFlow for optimal CPU performance
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=96
export OMP_NUM_THREADS=96
export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1

# For Intel CPUs (c5 instances use Intel CPUs)
export TF_ENABLE_ONEDNN_OPTS=1

# Disable CUDA to ensure CPU usage
export CUDA_VISIBLE_DEVICES=""
CPUEOF

chmod +x cpu_setup.sh

# SSH into the instance and run setup commands
echo "Setting up the instance environment..."
scp -i "$KEY_NAME.pem" -o StrictHostKeyChecking=no cpu_setup.sh \$USER_ACCOUNT@$INSTANCE_IP:~/

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

# Modify train-on-aws.sh to source the CPU optimizations
sed -i '1a source ~/cpu_setup.sh' train-on-aws.sh

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

# Create a script to monitor CPU usage
cat > monitor_cpu.sh << EOF3
#!/bin/bash
ssh -i "$KEY_NAME.pem" -o StrictHostKeyChecking=no \$USER_ACCOUNT@$INSTANCE_IP "top -b -n 1"
EOF3
chmod +x monitor_cpu.sh

# Create a script to download results
cat > download_results.sh << EOF4
#!/bin/bash
mkdir -p results
scp -i "$KEY_NAME.pem" -r \$USER_ACCOUNT@$INSTANCE_IP:~/emotion_training/models/branched_6class results/
echo "Results downloaded to results/branched_6class"
EOF4
chmod +x download_results.sh

# Create a script to stop/terminate the instance
cat > stop_instance.sh << EOF5
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
        echo "Instance stopped. To restart it later use: aws ec2 start-instances --instance-ids $INSTANCE_ID"
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
EOF5
chmod +x stop_instance.sh
SETUPEOF
chmod +x setup_instance.sh

# Execute the setup script
echo -e "${YELLOW}Executing setup script...${NC}"
./setup_instance.sh

echo -e "${GREEN}Setup complete!${NC}"
echo -e "${BLUE}Instance Type: $INSTANCE_TYPE${NC}"
echo -e "${BLUE}Instance ID: $INSTANCE_ID${NC}"
echo -e "${BLUE}Instance IP: $INSTANCE_IP${NC}"
echo -e "${BLUE}SSH Key: $KEY_NAME.pem${NC}"
echo ""
echo -e "${YELLOW}Available commands:${NC}"
echo "- ./check_progress.sh - Check training progress"
echo "- ./monitor_cpu.sh - Monitor CPU utilization in real-time"
echo "- ./download_results.sh - Download trained models"
echo "- ./stop_instance.sh - Stop or terminate the instance when done"
echo ""
echo -e "${RED}IMPORTANT: Remember to stop or terminate your instance when done to avoid unnecessary charges.${NC}"
echo -e "${RED}Hourly cost for $INSTANCE_TYPE is approximately \$4.08/hour.${NC}"
echo ""
echo -e "${YELLOW}Note: For future use, you can request GPU instance quotas at:${NC}"
echo -e "${BLUE}https://console.aws.amazon.com/servicequotas/home#!/services/ec2/quotas${NC}"
