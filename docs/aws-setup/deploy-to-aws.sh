#!/bin/bash
# AWS Deployment Script for Emotion Recognition Training
# This script generates all necessary commands to deploy and run training on AWS

# Colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Configuration
AWS_KEY=""
AWS_INSTANCE_IP=""
INSTANCE_TYPE=""

# Banner
echo -e "${BLUE}==========================================================================${NC}"
echo -e "${BLUE}                 AWS EMOTION RECOGNITION TRAINING SETUP                   ${NC}"
echo -e "${BLUE}==========================================================================${NC}"
echo ""

# Collect information
echo -e "${YELLOW}Please provide information about your AWS instance:${NC}"
echo ""

# Request EC2 key file path
read -p "Enter the path to your AWS key file (.pem): " AWS_KEY
if [ ! -f "$AWS_KEY" ]; then
    echo -e "${RED}Error: Key file not found at $AWS_KEY${NC}"
    exit 1
fi
chmod 400 "$AWS_KEY" 2>/dev/null
echo -e "${GREEN}Key file verified.${NC}"
echo ""

# Request EC2 instance IP
read -p "Enter your EC2 instance IP address: " AWS_INSTANCE_IP
echo ""

# Request instance type
echo "Select your EC2 instance type:"
echo "1) g4dn.xlarge (Standard GPU, ~$0.52/hour)"
echo "2) r5.2xlarge (High RAM, ~$0.50/hour)"
echo "3) p3.8xlarge (High Performance, 4 V100 GPUs, ~$12.24/hour)"
read -p "Enter choice [1-3]: " instance_choice

case $instance_choice in
    1) INSTANCE_TYPE="g4dn.xlarge";;
    2) INSTANCE_TYPE="r5.2xlarge";;
    3) INSTANCE_TYPE="p3.8xlarge";;
    *) echo -e "${RED}Invalid choice, defaulting to g4dn.xlarge${NC}"; INSTANCE_TYPE="g4dn.xlarge";;
esac
echo -e "${GREEN}Selected instance type: $INSTANCE_TYPE${NC}"
echo ""

# Request training parameters
read -p "Learning rate [default: 1e-5]: " LEARNING_RATE
LEARNING_RATE=${LEARNING_RATE:-1e-5}

read -p "Number of epochs [default: 50]: " EPOCHS
EPOCHS=${EPOCHS:-50}

read -p "Batch size [default: 32]: " BATCH_SIZE
BATCH_SIZE=${BATCH_SIZE:-32}

echo ""
echo -e "${GREEN}Training parameters set:${NC}"
echo "Learning rate: $LEARNING_RATE"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo ""

# Create a deployment directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DEPLOY_DIR="aws_deploy_$TIMESTAMP"
mkdir -p $DEPLOY_DIR

# Create upload script
cat > $DEPLOY_DIR/01_upload_to_aws.sh << EOF
#!/bin/bash
# Upload files to AWS instance

echo "Uploading files to AWS instance at $AWS_INSTANCE_IP..."
scp -i "$AWS_KEY" ../aws-emotion-recognition-training.tar.gz ec2-user@$AWS_INSTANCE_IP:~/
echo "Upload complete."
EOF
chmod +x $DEPLOY_DIR/01_upload_to_aws.sh

# Create setup script
cat > $DEPLOY_DIR/02_setup_on_aws.sh << EOF
#!/bin/bash
# Set up environment on AWS instance

echo "Setting up environment on AWS instance..."
ssh -i "$AWS_KEY" ec2-user@$AWS_INSTANCE_IP << 'ENDSSH'
# Extract files
echo "Extracting files..."
mkdir -p emotion_training
tar -xzf aws-emotion-recognition-training.tar.gz -C emotion_training
cd emotion_training

# Make scripts executable
chmod +x aws-instance-setup.sh train-on-aws.sh

# Set up environment
echo "Setting up environment..."
./aws-instance-setup.sh
ENDSSH

echo "Setup complete."
EOF
chmod +x $DEPLOY_DIR/02_setup_on_aws.sh

# Create training script
cat > $DEPLOY_DIR/03_run_training.sh << EOF
#!/bin/bash
# Run training on AWS instance

echo "Starting training on AWS instance ($INSTANCE_TYPE)..."
ssh -i "$AWS_KEY" ec2-user@$AWS_INSTANCE_IP << ENDSSH
cd emotion_training
./train-on-aws.sh --learning-rate $LEARNING_RATE --epochs $EPOCHS --batch-size $BATCH_SIZE
ENDSSH

echo "Training command sent. Check the AWS console for progress."
EOF
chmod +x $DEPLOY_DIR/03_run_training.sh

# Create download results script
cat > $DEPLOY_DIR/04_download_results.sh << EOF
#!/bin/bash
# Download results from AWS instance

echo "Downloading results from AWS instance..."
mkdir -p results
scp -i "$AWS_KEY" -r ec2-user@$AWS_INSTANCE_IP:~/emotion_training/models/branched_6class results/
echo "Results downloaded to results/branched_6class"
EOF
chmod +x $DEPLOY_DIR/04_download_results.sh

# Create one-step script to run all steps
cat > $DEPLOY_DIR/run_all_steps.sh << EOF
#!/bin/bash
# Run all deployment steps

# Upload files
./01_upload_to_aws.sh
echo ""

# Setup on AWS
./02_setup_on_aws.sh
echo ""

# Run training
./03_run_training.sh
echo ""

echo "Deployment complete. When training finishes, run ./04_download_results.sh to get the results."
EOF
chmod +x $DEPLOY_DIR/run_all_steps.sh

# Create a README
cat > $DEPLOY_DIR/README.txt << EOF
AWS DEPLOYMENT SCRIPTS
=====================

These scripts will help you deploy and run your emotion recognition training on AWS.

Steps:
1. Run ./01_upload_to_aws.sh to upload files to your AWS instance
2. Run ./02_setup_on_aws.sh to set up the environment on AWS
3. Run ./03_run_training.sh to start the training process
4. After training completes, run ./04_download_results.sh to download the results

Or simply run ./run_all_steps.sh to execute steps 1-3 in sequence.

Configuration:
- EC2 Instance: $AWS_INSTANCE_IP ($INSTANCE_TYPE)
- Training parameters:
  * Learning rate: $LEARNING_RATE
  * Epochs: $EPOCHS
  * Batch size: $BATCH_SIZE

IMPORTANT: Remember to stop or terminate your AWS instance when training is complete
to avoid unnecessary charges.
EOF

echo -e "${GREEN}Deployment scripts created in $DEPLOY_DIR/${NC}"
echo -e "${YELLOW}To deploy to AWS, navigate to $DEPLOY_DIR and follow these steps:${NC}"
echo "1. Run ./01_upload_to_aws.sh to upload files to your AWS instance"
echo "2. Run ./02_setup_on_aws.sh to set up the environment on AWS"
echo "3. Run ./03_run_training.sh to start the training process"
echo "4. After training completes, run ./04_download_results.sh to download the results"
echo ""
echo -e "${BLUE}Or simply run ./run_all_steps.sh to execute steps 1-3 in sequence.${NC}"
echo ""
echo -e "${RED}IMPORTANT: Remember to stop or terminate your AWS instance when training is complete${NC}"
echo -e "${RED}           to avoid unnecessary charges.${NC}"
echo ""
