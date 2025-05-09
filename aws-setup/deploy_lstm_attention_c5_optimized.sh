#!/bin/bash
# Deploy and train the LSTM attention model with only essential files:
# 1. Key Python scripts (train_branched_attention.py, sequence_data_generator.py, utils.py)
# 2. Required dependencies (requirements.txt)
# 3. FaceNet feature data (ravdess_features_facenet/, crema_d_features_facenet/)
# Optimized for c5.24xlarge CPU instance with 96 vCPUs

# Set script to exit on error
set -e

# ANSI colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Define variables
INSTANCE_TYPE="c5.24xlarge"  # High performance CPU instance with 96 vCPUs
INSTANCE_NAME="emotion-recognition-lstm-attention-model"
KEY_NAME="emotion-recognition-key-$(date +%Y%m%d%H%M%S)"
REGION="us-east-1"
AMI_ID="ami-0fe472d8a85bc7b0e"  # Amazon Linux 2
SECURITY_GROUP="emotion-recognition-sg"
LOG_FILE="training_lstm_attention_model.log"

# Print banner
echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}    DEPLOYING LSTM ATTENTION MODEL TO AWS (C5.24XLARGE)${NC}"
echo -e "${BLUE}    - With BiLSTM Temporal Modeling + Masking${NC}"
echo -e "${BLUE}    - With Attention Mechanism${NC}"
echo -e "${BLUE}    - With Focal Loss + Class Weighting${NC}"
echo -e "${BLUE}    - OPTIMIZED FOR CPU - ESSENTIAL FILES ONLY${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo ""

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}AWS CLI is not installed. Please install it first.${NC}"
    exit 1
fi

# Create directory for temporary files
mkdir -p tmp

# Create a list of essential files for training
echo -e "${YELLOW}Creating list of essential files...${NC}"
cat > tmp/essential_files.txt << EOF
scripts/train_branched_attention.py
scripts/sequence_data_generator.py
scripts/utils.py
requirements.txt
ravdess_features_facenet/*
crema_d_features_facenet/*
EOF

# Create a dedicated directory for the essential files
echo -e "${YELLOW}Creating directory for essential files...${NC}"
mkdir -p tmp/lstm_attention_train

# Copy essential Python scripts
echo -e "${YELLOW}Copying essential script files...${NC}"
cp scripts/train_branched_attention.py tmp/lstm_attention_train/
cp scripts/sequence_data_generator.py tmp/lstm_attention_train/
cp scripts/utils.py tmp/lstm_attention_train/
cp requirements.txt tmp/lstm_attention_train/

# Create directories for feature data
mkdir -p tmp/lstm_attention_train/ravdess_features_facenet
mkdir -p tmp/lstm_attention_train/crema_d_features_facenet

# Count files in each feature directory
RAVDESS_COUNT=$(find ravdess_features_facenet -name "*.npz" | wc -l)
CREMA_D_COUNT=$(find crema_d_features_facenet -name "*.npz" | wc -l)

echo -e "${YELLOW}Found ${RAVDESS_COUNT} RAVDESS feature files and ${CREMA_D_COUNT} CREMA-D feature files${NC}"

# Copy a subset of feature files for testing (optional)
# Uncomment if you want to only use a subset for testing
# echo -e "${YELLOW}Copying subset of feature files for testing...${NC}"
# find ravdess_features_facenet -name "*.npz" | head -50 | xargs -I{} cp {} tmp/lstm_attention_train/ravdess_features_facenet/
# find crema_d_features_facenet -name "*.npz" | head -50 | xargs -I{} cp {} tmp/lstm_attention_train/crema_d_features_facenet/

# Copy all feature files (comment out if using subset above)
echo -e "${YELLOW}Copying all feature files (this may take a while)...${NC}"
cp -r ravdess_features_facenet/* tmp/lstm_attention_train/ravdess_features_facenet/
cp -r crema_d_features_facenet/* tmp/lstm_attention_train/crema_d_features_facenet/

# Create archive with only essential files
echo -e "${YELLOW}Creating optimized archive...${NC}"
cd tmp
tar -czf lstm_attention_optimized.tar.gz lstm_attention_train
cd ..

# Check archive size
ARCHIVE_SIZE=$(du -h tmp/lstm_attention_optimized.tar.gz | cut -f1)
echo -e "${GREEN}Optimized archive created: ${ARCHIVE_SIZE}${NC}"

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
echo -e "${YELLOW}Launching EC2 instance (c5.24xlarge)...${NC}"
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

# Wait for SSH to be available with timeout
SSH_CONNECT_TIMEOUT=300  # 5 minutes timeout
SSH_CONNECT_INTERVAL=10  # 10 seconds interval
SSH_CONNECT_COUNT=$((SSH_CONNECT_TIMEOUT / SSH_CONNECT_INTERVAL))

for i in $(seq 1 $SSH_CONNECT_COUNT); do
    if ssh -i aws-setup/$KEY_NAME.pem -o StrictHostKeyChecking=no -o ConnectTimeout=5 ec2-user@$INSTANCE_IP echo "SSH connection established" &> /dev/null; then
        echo -e "${GREEN}SSH connection established.${NC}"
        break
    fi

    echo -e "${YELLOW}Waiting for SSH connection... Attempt $i of $SSH_CONNECT_COUNT${NC}"

    if [ $i -eq $SSH_CONNECT_COUNT ]; then
        echo -e "${RED}Failed to establish SSH connection after $SSH_CONNECT_TIMEOUT seconds.${NC}"
        echo -e "${YELLOW}Instance ID: $INSTANCE_ID${NC}"
        echo -e "${YELLOW}Instance IP: $INSTANCE_IP${NC}"
        echo -e "${YELLOW}You can try to connect manually: ssh -i aws-setup/$KEY_NAME.pem ec2-user@$INSTANCE_IP${NC}"
        exit 1
    fi

    sleep $SSH_CONNECT_INTERVAL
done

# Copy optimized project files to instance
echo -e "${YELLOW}Copying optimized files to instance...${NC}"
scp -i aws-setup/$KEY_NAME.pem -o StrictHostKeyChecking=no tmp/lstm_attention_optimized.tar.gz ec2-user@$INSTANCE_IP:~/lstm_attention_optimized.tar.gz

echo -e "${GREEN}Files copied.${NC}"

# Set up the instance and start training
echo -e "${YELLOW}Setting up the environment and starting training...${NC}"
ssh -i aws-setup/$KEY_NAME.pem -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP << EOF
    # Create directory and extract files
    mkdir -p ~/emotion_training
    tar -xzf lstm_attention_optimized.tar.gz -C ~/
    mv ~/lstm_attention_train/* ~/emotion_training/
    
    # Install dependencies
    sudo yum update -y
    sudo yum install -y python3 python3-pip git tmux htop
    
    # Configure pip and install dependencies
    pip3 install --user --upgrade pip
    
    # Install specific Python packages - using CPU version of TF
    cd ~/emotion_training
    pip3 install --user -r requirements.txt
    pip3 install --user tensorflow==2.4.0 scipy matplotlib pandas scikit-learn h5py
    
    # Force older urllib3 to avoid OpenSSL conflicts
    pip3 install --user urllib3==1.26.6
    
    # Create specific directories for model output
    mkdir -p ~/emotion_training/models/attention_focal_loss
    
    # Check CPU resources
    echo "Checking CPU resources..."
    python3 -c "import multiprocessing; print('CPU Cores Available:', multiprocessing.cpu_count())"
    
    # Create training script
    cat > ~/emotion_training/run_lstm_training.sh << 'EOFINNER'
#!/bin/bash
cd ~/emotion_training
echo "Starting LSTM attention model training at \$(date)"
# Force TensorFlow to use all available cores
export TF_INTRA_OP_PARALLELISM_THREADS=96
export TF_INTER_OP_PARALLELISM_THREADS=96
python3 train_branched_attention.py > training_lstm_attention_model.log 2>&1
echo "Training completed with exit code \$? at \$(date)"
EOFINNER

    chmod +x ~/emotion_training/run_lstm_training.sh
    
    # Start training in tmux session
    tmux new-session -d -s training "cd ~/emotion_training && ./run_lstm_training.sh"
    
    echo "LSTM attention model training started in tmux session."
    echo "Use 'tmux attach -t training' to view the training session."
    echo "Log file: ~/emotion_training/training_lstm_attention_model.log"
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
echo -e "ssh -i aws-setup/$KEY_NAME.pem ec2-user@$INSTANCE_IP 'tail -f ~/emotion_training/training_lstm_attention_model.log'"
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
LOG_FILE=training_lstm_attention_model.log
EOL

echo -e "${GREEN}Connection details saved to aws-setup/lstm_attention_model_connection.txt${NC}"

# Create monitoring scripts
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

echo -e "${GREEN}Created monitoring script: aws-setup/monitor_lstm_attention_model.sh${NC}"
echo -e "${YELLOW}Run this script to monitor the training progress.${NC}"
echo -e "${YELLOW}Use 'aws-setup/monitor_lstm_attention_model.sh --cpu' to monitor CPU usage.${NC}"
echo -e "${GREEN}Created download script: aws-setup/download_lstm_attention_results.sh${NC}"
echo -e "${YELLOW}Run this script when training is complete to download results.${NC}"
