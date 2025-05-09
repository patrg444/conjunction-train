#!/bin/bash
# Deploy and train the improved emotion recognition model with:
# 1. BiLSTM Temporal Modeling with proper masking
# 2. Attention Mechanism with explicit mask handling
# 3. Focal Loss with class weighting
# 4. Audio Data Augmentation

# Set script to exit on error
set -e

# ANSI colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Define variables
INSTANCE_TYPE="c5.24xlarge"  # High performance CPU instance with 96 vCPUs
INSTANCE_NAME="emotion-recognition-lstm-attention-model"
KEY_NAME="emotion-recognition-key-$(date +%Y%m%d%H%M%S)"
REGION="us-east-1"
AMI_ID="ami-0fe472d8a85bc7b0e"  # Amazon Linux 2 with NVIDIA drivers
SECURITY_GROUP="emotion-recognition-sg"
LOG_FILE="training_lstm_attention_model.log"

# Print banner
echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}    DEPLOYING IMPROVED LSTM EMOTION RECOGNITION MODEL TO AWS${NC}"
echo -e "${BLUE}    - With BiLSTM Temporal Modeling + Masking${NC}"
echo -e "${BLUE}    - With Attention Mechanism${NC}"
echo -e "${BLUE}    - With Focal Loss + Class Weighting${NC}"
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
    sudo yum install -y python3 python3-pip git tmux htop

    # Configure pip to use the latest version
    pip3 install --user --upgrade pip
    
    # Install Python packages with specific versions compatible with GPU
    cd ~/emotion_training
    pip3 install --user -r requirements.txt
    pip3 install --user tensorflow-gpu==2.18.0 scipy matplotlib
    pip3 install --user pandas scikit-learn h5py
    
    # Verify CPU cores available to TensorFlow
    echo "Checking CPU resources..."
    python3 -c "import tensorflow as tf; import multiprocessing; print('CPU Cores Available:', multiprocessing.cpu_count()); print('TensorFlow Devices:', tf.config.list_physical_devices())"

    # Create specific directories for model output if they don't exist
    mkdir -p ~/emotion_training/models/attention_focal_loss

    # Start training in a tmux session so it continues running after disconnection
    # First create a shell script to execute training
    cat > ~/emotion_training/run_lstm_training.sh << 'EOFINNER'
#!/bin/bash
cd ~/emotion_training
echo "Starting LSTM attention model training at \$(date)"
python3 scripts/train_branched_attention.py > $LOG_FILE 2>&1
echo "Training completed with exit code \$? at \$(date)"
EOFINNER

    chmod +x ~/emotion_training/run_lstm_training.sh

    # Start in tmux
    tmux new-session -d -s training "cd ~/emotion_training && ./run_lstm_training.sh"
    
    echo "LSTM attention model training started in tmux session."
    echo "Use 'tmux attach -t training' to view the training session."
    echo "Log file: ~/emotion_training/$LOG_FILE"
    echo "Monitor GPU usage: nvidia-smi"
EOF

echo -e "${GREEN}Setup complete and training started.${NC}"
echo ""
echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}                      DEPLOYMENT SUMMARY${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "${YELLOW}Instance ID:${NC} $INSTANCE_ID"
echo -e "${YELLOW}Instance Type:${NC} $INSTANCE_TYPE (High-performance CPU with 96 vCPUs)"
echo -e "${YELLOW}Instance IP:${NC} $INSTANCE_IP"
echo -e "${YELLOW}SSH Key:${NC} aws-setup/$KEY_NAME.pem"
echo ""
echo -e "${YELLOW}To connect to the instance:${NC}"
echo -e "ssh -i aws-setup/$KEY_NAME.pem ec2-user@$INSTANCE_IP"
echo ""
echo -e "${YELLOW}To monitor training:${NC}"
echo -e "ssh -i aws-setup/$KEY_NAME.pem ec2-user@$INSTANCE_IP 'tail -f ~/emotion_training/$LOG_FILE'"
echo ""
echo -e "${YELLOW}To check CPU usage:${NC}"
echo -e "ssh -i aws-setup/$KEY_NAME.pem ec2-user@$INSTANCE_IP 'top -b -n 1 | head -20'"
echo ""
echo -e "${YELLOW}To download results when training completes:${NC}"
echo -e "scp -i aws-setup/$KEY_NAME.pem -r ec2-user@$INSTANCE_IP:~/emotion_training/models/attention_focal_loss ."
echo -e "${BLUE}=================================================================${NC}"

# Save connection details for future use
cat > aws-setup/lstm_attention_model_connection.txt << EOL
INSTANCE_ID=$INSTANCE_ID
INSTANCE_IP=$INSTANCE_IP
KEY_NAME=$KEY_NAME
KEY_FILE=aws-setup/$KEY_NAME.pem
LOG_FILE=$LOG_FILE
EOL

echo -e "${GREEN}Connection details saved to aws-setup/lstm_attention_model_connection.txt${NC}"

# Create a monitoring script
cat > aws-setup/monitor_lstm_attention_model.sh << EOL
#!/bin/bash
# Script to monitor training progress of the LSTM attention model

# Source connection details
source aws-setup/lstm_attention_model_connection.txt

# ANSI colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "\${BLUE}=================================================================\${NC}"
echo -e "\${BLUE}     MONITORING LSTM ATTENTION MODEL TRAINING                    \${NC}"
echo -e "\${BLUE}=================================================================\${NC}"
echo -e "\${YELLOW}Instance:\${NC} \$INSTANCE_IP"
echo -e "\${YELLOW}Log file:\${NC} ~/emotion_training/\$LOG_FILE"
echo -e "\${GREEN}Streaming training log...\${NC}"
echo -e "\${BLUE}=================================================================\${NC}"
echo ""

# Option for CPU monitoring
if [ "\$1" == "--cpu" ]; then
    echo -e "\${YELLOW}Monitoring CPU usage...\${NC}"
    ssh -i \$KEY_FILE -o StrictHostKeyChecking=no ec2-user@\$INSTANCE_IP "while true; do clear; top -b -n 1 | head -20; sleep 5; done"
    exit 0
fi

# Stream the log
ssh -i \$KEY_FILE -o StrictHostKeyChecking=no ec2-user@\$INSTANCE_IP "tail -f ~/emotion_training/\$LOG_FILE"
EOL

chmod +x aws-setup/monitor_lstm_attention_model.sh

echo -e "${GREEN}Created monitoring script: aws-setup/monitor_lstm_attention_model.sh${NC}"
echo -e "${YELLOW}Run this script to monitor the training progress.${NC}"
echo -e "${YELLOW}Use 'aws-setup/monitor_lstm_attention_model.sh --cpu' to monitor CPU usage.${NC}"
echo ""
echo -e "${CYAN}For advanced continuous monitoring with live feed:${NC}"
echo -e "${YELLOW}Use 'aws-setup/live_continuous_monitor.sh' with the following options:${NC}"
echo -e "  - No arguments: Live training log feed"
echo -e "  - 'gpu': Live GPU metrics monitoring"
echo -e "  - 'system': Live system resource monitoring"
echo -e "  - 'all': Comprehensive dashboard with all metrics in tmux panes"

# Create a script to download results
cat > aws-setup/download_lstm_attention_results.sh << EOL
#!/bin/bash
# Script to download the LSTM attention model results

# Source connection details
source aws-setup/lstm_attention_model_connection.txt

# ANSI colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

DOWNLOAD_DIR="lstm_attention_model_results"

echo -e "\${BLUE}=================================================================\${NC}"
echo -e "\${BLUE}     DOWNLOADING LSTM ATTENTION MODEL RESULTS                    \${NC}"
echo -e "\${BLUE}=================================================================\${NC}"

# Create download directory
mkdir -p \$DOWNLOAD_DIR
echo -e "\${YELLOW}Downloading trained model and logs to \$DOWNLOAD_DIR...\${NC}"

# Download the models and logs
scp -i \$KEY_FILE -r ec2-user@\$INSTANCE_IP:~/emotion_training/models/attention_focal_loss \$DOWNLOAD_DIR/
scp -i \$KEY_FILE ec2-user@\$INSTANCE_IP:~/emotion_training/\$LOG_FILE \$DOWNLOAD_DIR/

echo -e "\${GREEN}Results downloaded to \$DOWNLOAD_DIR directory${NC}"
echo -e "\${BLUE}=================================================================\${NC}"
EOL

chmod +x aws-setup/download_lstm_attention_results.sh

echo -e "${GREEN}Created download script: aws-setup/download_lstm_attention_results.sh${NC}"
echo -e "${YELLOW}Run this script when training is complete to download results.${NC}"
