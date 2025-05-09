#!/bin/bash
# Script to download trained model and results from EC2 instance

# ANSI colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Instance details
INSTANCE_ID="i-0dd2f787db00b205f"
INSTANCE_IP="98.82.121.48"
KEY_FILE="aws-setup/emotion-recognition-key-20250322082227.pem"

echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}     DOWNLOADING LSTM ATTENTION MODEL RESULTS                   ${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "${YELLOW}Instance ID:${NC} $INSTANCE_ID"
echo -e "${YELLOW}Instance IP:${NC} $INSTANCE_IP"
echo -e "${BLUE}=================================================================${NC}"
echo ""

# Create local directories for results
echo -e "${YELLOW}Creating local directories for results...${NC}"
mkdir -p model_results/attention_focal_loss
mkdir -p model_results/logs
mkdir -p model_results/visualizations

# Check if instance is available
echo -e "${YELLOW}Checking if the instance is available...${NC}"
if ! ssh -i $KEY_FILE -o StrictHostKeyChecking=no -o ConnectTimeout=5 ec2-user@$INSTANCE_IP echo "SSH connection established" &> /dev/null; then
    echo -e "${RED}Failed to connect to the instance. Please check if it's running.${NC}"
    exit 1
fi
echo -e "${GREEN}Instance is available.${NC}"

# Check if the model files exist
echo -e "${YELLOW}Checking for model files...${NC}"
if ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "[ ! -f ~/emotion_training/models/attention_focal_loss/final_model.h5 ]"; then
    echo -e "${YELLOW}Final model file not found. Training may still be in progress.${NC}"
    echo -e "${YELLOW}Checking for latest checkpoint model file...${NC}"
    
    # Look for checkpoint files
    if ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "[ -d ~/emotion_training/models/attention_focal_loss ]"; then
        echo -e "${YELLOW}Found model directory. Checking for latest checkpoint...${NC}"
        LATEST_CHECKPOINT=$(ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "find ~/emotion_training/models/attention_focal_loss -name '*.h5' | sort -r | head -1")
        
        if [ -z "$LATEST_CHECKPOINT" ]; then
            echo -e "${RED}No model checkpoints found. Training may not have saved any models yet.${NC}"
        else
            echo -e "${GREEN}Found checkpoint: $LATEST_CHECKPOINT${NC}"
            echo -e "${YELLOW}Downloading latest checkpoint model...${NC}"
            scp -i $KEY_FILE ec2-user@$INSTANCE_IP:"$LATEST_CHECKPOINT" model_results/attention_focal_loss/
        fi
    else
        echo -e "${RED}Model directory not found. Training may not have started properly.${NC}"
    fi
else
    echo -e "${GREEN}Final model file found. Downloading...${NC}"
    scp -i $KEY_FILE ec2-user@$INSTANCE_IP:~/emotion_training/models/attention_focal_loss/final_model.h5 model_results/attention_focal_loss/
fi

# Download training logs
echo -e "${YELLOW}Downloading training logs...${NC}"
ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "find ~/emotion_training -name '*.log' | xargs ls -t | head -5" | while read log_file; do
    echo -e "${YELLOW}Downloading log: $log_file${NC}"
    scp -i $KEY_FILE ec2-user@$INSTANCE_IP:"$log_file" model_results/logs/
done

# Download any visualizations or metrics
echo -e "${YELLOW}Checking for visualization files...${NC}"
ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "find ~/emotion_training -name '*.png' -o -name '*.jpg' -o -name '*.svg' -o -name '*.csv' | grep -v 'datasets'" | while read vis_file; do
    echo -e "${YELLOW}Downloading visualization: $vis_file${NC}"
    # Create the directory structure locally
    local_dir="model_results/visualizations/$(dirname "$vis_file" | sed 's/.*emotion_training\///')"
    mkdir -p "$local_dir"
    scp -i $KEY_FILE ec2-user@$INSTANCE_IP:"$vis_file" "$local_dir/"
done

# Download model history JSON if exists
echo -e "${YELLOW}Checking for training history...${NC}"
if ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP "[ -f ~/emotion_training/models/attention_focal_loss/history.json ]"; then
    echo -e "${GREEN}Found training history. Downloading...${NC}"
    scp -i $KEY_FILE ec2-user@$INSTANCE_IP:~/emotion_training/models/attention_focal_loss/history.json model_results/attention_focal_loss/
fi

echo -e "${GREEN}Results downloaded to model_results/ directory.${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}                      DOWNLOAD SUMMARY${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "${YELLOW}Model files:${NC}"
find model_results/attention_focal_loss -type f | sort

echo -e "\n${YELLOW}Log files:${NC}"
find model_results/logs -type f | sort

echo -e "\n${YELLOW}Visualization files:${NC}"
find model_results/visualizations -type f | sort
echo -e "${BLUE}=================================================================${NC}"
