#!/bin/bash
# Script to deploy the LSTM attention model WITHOUT audio augmentation to EC2
# This uses a c5.24xlarge instance for optimal CPU performance

# ANSI color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

set -e # Exit on error

# Display header
echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}     DEPLOYMENT OF LSTM ATTENTION MODEL (NO AUGMENTATION)        ${NC}"
echo -e "${BLUE}=================================================================${NC}"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}Error: AWS CLI is not installed. Please install it first.${NC}"
    exit 1
fi

# Check if AWS is configured
if ! aws configure list &> /dev/null; then
    echo -e "${RED}Error: AWS CLI is not configured. Please run 'aws configure' first.${NC}"
    exit 1
fi

# Instance type and configuration
INSTANCE_TYPE="c5.24xlarge" # 96 vCPUs
KEY_NAME="emotion-recognition-key-lstm-attention-no-aug"
TIMESTAMP=$(date +%Y%m%d%H%M%S)
KEY_FILE="aws-setup/emotion-recognition-key-${TIMESTAMP}.pem"
AMI_ID="ami-0440d3b780d96b29d" # Amazon Linux 2 AMI
SECURITY_GROUP="EmotionRecognitionTraining"
VOLUME_SIZE=8 # GB

echo -e "${YELLOW}Creating EC2 instance...${NC}"
echo "Instance type: $INSTANCE_TYPE"
echo "AMI: $AMI_ID"
echo "Key pair: $KEY_NAME"

# Create key pair
aws ec2 create-key-pair --key-name $KEY_NAME --query 'KeyMaterial' --output text > $KEY_FILE
chmod 400 $KEY_FILE

# Check if security group exists
if ! aws ec2 describe-security-groups --group-names $SECURITY_GROUP &> /dev/null; then
    echo -e "${YELLOW}Creating security group...${NC}"
    aws ec2 create-security-group --group-name $SECURITY_GROUP --description "Security group for Emotion Recognition training"
    aws ec2 authorize-security-group-ingress --group-name $SECURITY_GROUP --protocol tcp --port 22 --cidr 0.0.0.0/0
fi

# Launch instance
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-groups $SECURITY_GROUP \
    --block-device-mappings "[{\"DeviceName\":\"/dev/xvda\",\"Ebs\":{\"VolumeSize\":$VOLUME_SIZE,\"DeleteOnTermination\":true}}]" \
    --count 1 \
    --output text \
    --query 'Instances[0].InstanceId')

echo -e "${GREEN}Instance created: $INSTANCE_ID${NC}"

# Wait for instance to be running
echo -e "${YELLOW}Waiting for instance to start...${NC}"
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# Get public IP
INSTANCE_IP=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --output text --query 'Reservations[0].Instances[0].PublicIpAddress')
echo -e "${GREEN}Instance is running at $INSTANCE_IP${NC}"

# Save connection details to be used by other scripts
echo "# LSTM Attention Model (No Augmentation) Instance Details" > aws-setup/lstm_attention_model_connection.txt
echo "INSTANCE_ID=\"$INSTANCE_ID\"" >> aws-setup/lstm_attention_model_connection.txt
echo "INSTANCE_IP=\"$INSTANCE_IP\"" >> aws-setup/lstm_attention_model_connection.txt
echo "KEY_FILE=\"$KEY_FILE\"" >> aws-setup/lstm_attention_model_connection.txt
echo "LOG_FILE=\"training_lstm_attention_no_aug.log\"" >> aws-setup/lstm_attention_model_connection.txt

echo -e "${YELLOW}Waiting for SSH to be available...${NC}"
while ! ssh -i $KEY_FILE -o StrictHostKeyChecking=no -o ConnectTimeout=5 ec2-user@$INSTANCE_IP echo "SSH is up" 2> /dev/null
do
    echo "Waiting for SSH to come up..."
    sleep 5
done

echo -e "${GREEN}SSH is up and running${NC}"

# Prepare local files for uploading
echo -e "${YELLOW}Preparing files for upload...${NC}"

# Create a temporary directory
TMP_DIR=$(mktemp -d)
mkdir -p $TMP_DIR/scripts
mkdir -p $TMP_DIR/ravdess_features_facenet
mkdir -p $TMP_DIR/crema_d_features_facenet
mkdir -p $TMP_DIR/models/attention_focal_loss_no_aug

# Copy only essential files to reduce transfer size
cp scripts/train_branched_attention_no_aug.py $TMP_DIR/scripts/
cp scripts/sequence_data_generator.py $TMP_DIR/scripts/

# Copy RAVDESS features (only a subset if needed for testing)
cp -r ravdess_features_facenet/* $TMP_DIR/ravdess_features_facenet/

# Copy CREMA-D features
cp -r crema_d_features_facenet/*.npz $TMP_DIR/crema_d_features_facenet/

# Create a requirements.txt file with pinned versions to avoid dependency issues
cat > $TMP_DIR/requirements.txt << EOF
tensorflow==2.6.0
numpy==1.19.5
scipy==1.7.0
scikit-learn==0.24.2
matplotlib==3.4.2
pandas==1.3.0
urllib3==1.26.6  # Pinned version to avoid OpenSSL issues
EOF

# Create the setup script to run on the EC2 instance
cat > $TMP_DIR/setup.sh << 'EOF'
#!/bin/bash
set -e

# Set up environment
echo "Setting up environment..."
cd ~
mkdir -p emotion_training
mkdir -p emotion_training/logs
mkdir -p emotion_training/models/attention_focal_loss_no_aug

# Install dependencies
echo "Installing dependencies..."
sudo yum update -y
sudo yum install -y python3 python3-devel python3-pip git

# Install Python packages - with specific versions to avoid conflicts
echo "Installing Python packages..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# Configure TensorFlow to use all CPUs effectively
echo "export TF_NUM_INTEROP_THREADS=96" >> ~/.bashrc
echo "export TF_NUM_INTRAOP_THREADS=96" >> ~/.bashrc
echo "export OMP_NUM_THREADS=96" >> ~/.bashrc
source ~/.bashrc

echo "Setup complete!"
EOF

# Create the training script to run on the EC2 instance
cat > $TMP_DIR/train.sh << 'EOF'
#!/bin/bash

# Start training and log output
cd ~/emotion_training
python3 scripts/train_branched_attention_no_aug.py > training_lstm_attention_no_aug.log 2>&1 &

echo "Training started in background with process ID: $!"
echo "You can monitor the logs with: tail -f training_lstm_attention_no_aug.log"
EOF

# Make scripts executable
chmod +x $TMP_DIR/setup.sh
chmod +x $TMP_DIR/train.sh

echo -e "${YELLOW}Uploading files to EC2 instance...${NC}"
echo "This may take a while depending on your internet connection..."

# Create a tar.gz archive of the temp directory to speed up transfer
TAR_FILE="/tmp/ec2-upload-${TIMESTAMP}.tar.gz"
cd $TMP_DIR
tar -czf $TAR_FILE .
cd - > /dev/null

# Upload the archive
scp -i $KEY_FILE -o StrictHostKeyChecking=no $TAR_FILE ec2-user@$INSTANCE_IP:~/ec2-upload.tar.gz

# Clean up local temp files
rm -rf $TMP_DIR
rm $TAR_FILE

# SSH into the instance to extract the files and set up the environment
echo -e "${YELLOW}Setting up the EC2 environment...${NC}"
ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP << 'ENDSSH'
mkdir -p ~/emotion_training
cd ~/emotion_training
tar -xzf ~/ec2-upload.tar.gz
rm ~/ec2-upload.tar.gz

# Execute the setup script
bash setup.sh

# Start the training
bash train.sh

echo "EC2 setup complete and training started!"
ENDSSH

echo -e "${GREEN}===============================================================${NC}"
echo -e "${GREEN}LSTM Attention Model (No Augmentation) deployment complete!${NC}"
echo -e "${GREEN}===============================================================${NC}"
echo -e "Instance ID: ${BLUE}$INSTANCE_ID${NC}"
echo -e "Public IP: ${BLUE}$INSTANCE_IP${NC}"
echo -e "SSH Key: ${BLUE}$KEY_FILE${NC}"
echo -e "${YELLOW}To connect:${NC} ssh -i $KEY_FILE ec2-user@$INSTANCE_IP"
echo -e "${YELLOW}To monitor training:${NC} ./aws-setup/continuous_training_monitor.sh"
echo -e "${GREEN}===============================================================${NC}"
