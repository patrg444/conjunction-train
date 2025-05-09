#!/bin/bash
# Script to download LSTM attention model (no augmentation) results from EC2

# ANSI color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Source connection details
source aws-setup/lstm_attention_model_connection.txt

# Create local directory for model
LOCAL_MODEL_DIR="models/attention_focal_loss_no_aug"
mkdir -p "$LOCAL_MODEL_DIR"

echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}     DOWNLOADING LSTM ATTENTION MODEL (NO AUGMENTATION)          ${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "${YELLOW}Instance IP:${NC} $INSTANCE_IP"
echo -e "${YELLOW}Key file:${NC} $KEY_FILE"
echo -e "${YELLOW}Local model directory:${NC} $LOCAL_MODEL_DIR"
echo ""

# Check if instance is running
echo -e "${YELLOW}Checking if instance is running...${NC}"
INSTANCE_STATUS=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null)
if [[ "$INSTANCE_STATUS" != "running" ]]; then
    echo -e "${RED}Error: Instance is not running (status: $INSTANCE_STATUS)${NC}"
    exit 1
fi

echo -e "${GREEN}Instance is running with status: $INSTANCE_STATUS${NC}"

# Test SSH connection
echo -e "${YELLOW}Testing SSH connection...${NC}"
if ! ssh -i $KEY_FILE -o StrictHostKeyChecking=no -o ConnectTimeout=5 ec2-user@$INSTANCE_IP "echo Connection successful" &>/dev/null; then
    echo -e "${RED}Error: SSH connection failed!${NC}"
    exit 1
fi

echo -e "${GREEN}SSH connection successful${NC}"

# Download model files
echo -e "${YELLOW}Downloading model files...${NC}"
ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP << 'EOT'
    mkdir -p ~/emotion_training/models/download_temp
    cd ~/emotion_training/models/attention_focal_loss_no_aug
    
    if [[ -f "model_best.keras" ]]; then
        cp model_best.keras ~/emotion_training/models/download_temp/
    else
        echo "Warning: Best model file not found"
    fi
    
    if [[ -f "final_model.keras" ]]; then
        cp final_model.keras ~/emotion_training/models/download_temp/
    else
        echo "Warning: Final model file not found"
    fi
    
    cd ~/emotion_training
    if [[ -f "training_lstm_attention_no_aug.log" ]]; then
        cp training_lstm_attention_no_aug.log models/download_temp/
    else
        echo "Warning: Training log file not found"
    fi
    
    cd ~/emotion_training/models
    tar -czf download_package.tar.gz download_temp/*
EOT

# Download the tarball
echo -e "${YELLOW}Downloading model package...${NC}"
scp -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP:~/emotion_training/models/download_package.tar.gz /tmp/

# Clean up remote temp files
ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "rm -rf ~/emotion_training/models/download_temp ~/emotion_training/models/download_package.tar.gz"

# Extract files locally
echo -e "${YELLOW}Extracting model files...${NC}"
tar -xzf /tmp/download_package.tar.gz -C $LOCAL_MODEL_DIR
rm /tmp/download_package.tar.gz

# Check if files were downloaded
if [[ -f "$LOCAL_MODEL_DIR/model_best.keras" ]]; then
    echo -e "${GREEN}Best model downloaded successfully${NC}"
else
    echo -e "${YELLOW}Warning: Best model was not downloaded. It may not be available yet.${NC}"
fi

if [[ -f "$LOCAL_MODEL_DIR/final_model.keras" ]]; then
    echo -e "${GREEN}Final model downloaded successfully${NC}"
else
    echo -e "${YELLOW}Warning: Final model was not downloaded. It may not be available yet.${NC}"
fi

if [[ -f "$LOCAL_MODEL_DIR/training_lstm_attention_no_aug.log" ]]; then
    echo -e "${GREEN}Training log downloaded successfully${NC}"
    # Extract key metrics
    echo -e "${BLUE}Training summary from log:${NC}"
    grep "Training history summary:" -A 5 "$LOCAL_MODEL_DIR/training_lstm_attention_no_aug.log" || echo "No training summary found in log yet"
else
    echo -e "${YELLOW}Warning: Training log was not downloaded. It may not be available yet.${NC}"
fi

echo -e "${GREEN}===============================================================${NC}"
echo -e "${GREEN}Download complete!${NC}"
echo -e "${GREEN}===============================================================${NC}"
echo -e "Files saved to: ${BLUE}$LOCAL_MODEL_DIR${NC}"
echo -e "${GREEN}===============================================================${NC}"
