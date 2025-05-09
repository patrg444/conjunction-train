#!/bin/bash
# AWS GPU Instance Fallback Script
# This script attempts to launch high-performance GPU instances in sequence,
# falling back to alternatives if the first choice is unavailable

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
echo -e "${BLUE}              HIGH-PERFORMANCE GPU INSTANCE SETUP                         ${NC}"
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

# Define GPU instance types to try (in order of preference)
GPU_INSTANCES=(
    "g4dn.24xlarge" 
    "g5.24xlarge" 
    "p4d.24xlarge" 
    "g4dn.16xlarge" 
    "g4dn.12xlarge" 
    "g4dn.8xlarge" 
    "g4dn.4xlarge" 
    "g4dn.2xlarge" 
    "g4dn.xlarge"
)

echo -e "${YELLOW}Will try to launch the following instance types in order:${NC}"
for instance in "${GPU_INSTANCES[@]}"; do
    echo "- $instance"
done
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
        --filters "Name=name,Values=*Deep Learning AMI GPU*" \
        --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
        --output text)
fi

if [ -z "$AMI_ID" ] || [ "$AMI_ID" = "None" ]; then
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

# Try to launch instances in order
INSTANCE_ID=""
INSTANCE_TYPE=""

for instance_type in "${GPU_INSTANCES[@]}"; do
    echo -e "${YELLOW}Attempting to launch instance of type ${BLUE}$instance_type${YELLOW}...${NC}"
    
    # Try to launch this instance type
    LAUNCH_OUTPUT=$(aws ec2 run-instances \
        --image-id "$AMI_ID" \
        --instance-type "$instance_type" \
        --key-name "$KEY_NAME" \
        --security-group-ids "$SECURITY_GROUP_ID" \
        --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":100,\"VolumeType\":\"gp2\"}}]" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=EmotionRecognitionTraining}]" \
        --query 'Instances[0].InstanceId' \
        --output text 2>&1)
        
    # Check if the launch was successful
    if [[ $LAUNCH_OUTPUT == i-* ]]; then
        INSTANCE_ID=$LAUNCH_OUTPUT
        INSTANCE_TYPE=$instance_type
        echo -e "${GREEN}Successfully launched $instance_type instance with ID: $INSTANCE_ID${NC}"
        break
    else
        echo -e "${RED}Failed to launch $instance_type: $LAUNCH_OUTPUT${NC}"
        echo -e "${YELLOW}Trying next instance type...${NC}"
        echo ""
    fi
done

if [ -z "$INSTANCE_ID" ]; then
    echo -e "${RED}Failed to launch any GPU instance. Please check your AWS account limits or try again later.${NC}"
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

# Create an optimized setup script for the instance based on the instance type
cat > gpu_setup.sh << 'GPUEOF'
#!/bin/bash
# GPU-specific optimizations for $INSTANCE_TYPE

# Configure TensorFlow for optimal GPU performance
if [[ "$INSTANCE_TYPE" == *"p4d"* ]]; then
  # p4d-specific optimizations (A100 GPUs)
  export TF_XLA_FLAGS="--tf_xla_auto_jit=2"
  export TF_GPU_THREAD_MODE=gpu_private
  export TF_GPU_THREAD_COUNT=8
  export TF_CUDA_COMPUTE_CAPABILITIES=8.0
elif [[ "$INSTANCE_TYPE" == *"g5"* ]]; then
  # g5-specific optimizations (A10G GPUs)
  export TF_XLA_FLAGS="--tf_xla_auto_jit=2"
  export TF_GPU_THREAD_MODE=gpu_private
  export TF_CUDA_COMPUTE_CAPABILITIES=8.6
else
  # g4dn-specific optimizations (T4 GPUs)
  export TF_XLA_FLAGS="--tf_xla_auto_jit=2"
  export TF_CUDA_COMPUTE_CAPABILITIES=7.5
fi

# Enable automatic mixed precision for faster training
export TF_ENABLE_AUTO_MIXED_PRECISION=1
GPUEOF

chmod +x gpu_setup.sh

# SSH into the instance and run setup commands
echo "Setting up the instance environment..."
scp -i "$KEY_NAME.pem" -o StrictHostKeyChecking=no gpu_setup.sh \$USER_ACCOUNT@$INSTANCE_IP:~/

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

# Modify train-on-aws.sh to source the GPU optimizations
sed -i '1a source ~/gpu_setup.sh' train-on-aws.sh

# Verify NVIDIA drivers and GPUs are available
echo "Verifying GPU setup..."
nvidia-smi

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

# Create a script to monitor GPU usage
cat > monitor_gpu.sh << EOF3
#!/bin/bash
ssh -i "$KEY_NAME.pem" -o StrictHostKeyChecking=no \$USER_ACCOUNT@$INSTANCE_IP "watch -n 1 nvidia-smi"
EOF3
chmod +x monitor_gpu.sh

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
EOF5
chmod +x stop_instance.sh
EOF
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
echo "- ./monitor_gpu.sh - Monitor GPU utilization in real-time"
echo "- ./download_results.sh - Download trained models"
echo "- ./stop_instance.sh - Stop or terminate the instance when done"
echo ""
echo -e "${RED}IMPORTANT: Remember to stop or terminate your instance when done to avoid unnecessary charges.${NC}"
echo -e "${RED}Hourly cost for $INSTANCE_TYPE varies between \$0.50 and \$38 depending on the instance type.${NC}"
