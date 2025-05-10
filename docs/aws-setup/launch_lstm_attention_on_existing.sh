#!/bin/bash
# Script to deploy LSTM attention model on existing instance
# This leverages the idle c5.24xlarge instance that's already running

# ANSI colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Existing instance details
INSTANCE_ID="i-0dd2f787db00b205f"
INSTANCE_IP="98.82.121.48"
KEY_FILE="aws-setup/emotion-recognition-key-20250322082227.pem"
LOG_FILE="training_lstm_attention_model.log"

echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}     DEPLOYING LSTM ATTENTION MODEL TO EXISTING INSTANCE         ${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "${YELLOW}Instance ID:${NC} $INSTANCE_ID"
echo -e "${YELLOW}Instance IP:${NC} $INSTANCE_IP"
echo -e "${YELLOW}Instance Type:${NC} c5.24xlarge (96 vCPUs)"
echo -e "${BLUE}=================================================================${NC}"
echo ""

# Check if instance is available
echo -e "${YELLOW}Checking if the instance is available...${NC}"
if ! ssh -i $KEY_FILE -o StrictHostKeyChecking=no -o ConnectTimeout=5 ec2-user@$INSTANCE_IP echo "SSH connection established" &> /dev/null; then
    echo -e "${RED}Failed to connect to the instance. Please check if it's running.${NC}"
    exit 1
fi
echo -e "${GREEN}Instance is available.${NC}"

# Check if sequence data generator is available
echo -e "${YELLOW}Checking for required training scripts...${NC}"
if ! ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "test -f ~/emotion_training/scripts/train_branched_attention.py && echo 'Found'"; then
    echo -e "${RED}Required script train_branched_attention.py not found on the instance.${NC}"
    
    # Upload from local if available
    if [ -f "scripts/train_branched_attention.py" ]; then
        echo -e "${YELLOW}Uploading train_branched_attention.py from local...${NC}"
        scp -i $KEY_FILE scripts/train_branched_attention.py ec2-user@$INSTANCE_IP:~/emotion_training/scripts/
    else
        echo -e "${RED}Can't find train_branched_attention.py locally either. Please check your setup.${NC}"
        exit 1
    fi
fi

# Check for sequence data generator
if ! ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "test -f ~/emotion_training/scripts/sequence_data_generator.py && echo 'Found'"; then
    echo -e "${RED}Required script sequence_data_generator.py not found on the instance.${NC}"
    
    # Upload from local if available
    if [ -f "scripts/sequence_data_generator.py" ]; then
        echo -e "${YELLOW}Uploading sequence_data_generator.py from local...${NC}"
        scp -i $KEY_FILE scripts/sequence_data_generator.py ec2-user@$INSTANCE_IP:~/emotion_training/scripts/
    else
        echo -e "${RED}Can't find sequence_data_generator.py locally either. Please check your setup.${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}All required training scripts are available.${NC}"

# Create model output directory
echo -e "${YELLOW}Creating model output directory...${NC}"
ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "mkdir -p ~/emotion_training/models/attention_focal_loss"

# Start training in a tmux session for persistence
echo -e "${YELLOW}Starting LSTM attention model training...${NC}"
ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP << EOF
    # Create a shell script to execute training
    cat > ~/emotion_training/run_lstm_attention.sh << 'EOFINNER'
#!/bin/bash
cd ~/emotion_training
echo "Starting LSTM attention model training at \$(date)"
python3 scripts/train_branched_attention.py > $LOG_FILE 2>&1
echo "Training completed with exit code \$? at \$(date)"
EOFINNER

    chmod +x ~/emotion_training/run_lstm_attention.sh

    # Start in tmux
    tmux new-session -d -s lstm_attention "cd ~/emotion_training && ./run_lstm_attention.sh"
    
    echo "LSTM attention model training started in tmux session."
EOF

echo -e "${GREEN}LSTM attention model training started on the existing instance.${NC}"
echo ""
echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}                      DEPLOYMENT SUMMARY${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "${YELLOW}Instance ID:${NC} $INSTANCE_ID"
echo -e "${YELLOW}Instance IP:${NC} $INSTANCE_IP"
echo -e "${YELLOW}SSH Key:${NC} $KEY_FILE"
echo ""
echo -e "${YELLOW}To monitor training:${NC}"
echo -e "bash aws-setup/stream_logs.sh"
echo ""
echo -e "${YELLOW}To check CPU usage:${NC}"
echo -e "bash aws-setup/monitor_cpu_usage.sh"
echo ""
echo -e "${YELLOW}To check training files:${NC}"
echo -e "bash aws-setup/check_training_files.sh"
echo -e "${BLUE}=================================================================${NC}"

# Create connection details file
cat > aws-setup/lstm_attention_model_connection.txt << EOL
INSTANCE_ID=$INSTANCE_ID
INSTANCE_IP=$INSTANCE_IP
KEY_FILE=$KEY_FILE
LOG_FILE=$LOG_FILE
EOL

echo -e "${GREEN}Connection details saved to aws-setup/lstm_attention_model_connection.txt${NC}"
