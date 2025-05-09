#!/bin/bash
# Deploy fixed CNN-LSTM model with corrected attention output shape
# Script to transfer the fixed model to EC2 and start training

# Colors for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# EC2 connection details
EC2_KEY="$HOME/Downloads/gpu-key.pem"
EC2_INSTANCE="ubuntu@54.162.134.77"
REMOTE_DIR="/home/ubuntu/emotion_project"

# Local files to transfer
LOCAL_MODEL_FILE="improved_cnn_lstm_model_fixed.py"

echo -e "${BLUE}Preparing to deploy fixed CNN-LSTM model with attention${NC}"
echo -e "${YELLOW}This model includes:${NC}"
echo -e "  - Fixed shape mismatch issue with GlobalAveragePooling1D"
echo -e "  - Attention mechanism for better sequence understanding"
echo -e "  - Stronger regularization (L2 + higher dropout)"
echo -e "  - Lower learning rate with improved scheduling"
echo -e "  - Stratified train/validation split"
echo -e "  - Class weighting to handle imbalance"

# Check if the key file exists
if [ ! -f "$EC2_KEY" ]; then
    echo -e "${RED}Error: SSH key not found at $EC2_KEY${NC}"
    exit 1
fi

# Check if model file exists
if [ ! -f "$LOCAL_MODEL_FILE" ]; then
    echo -e "${RED}Error: Model file not found: $LOCAL_MODEL_FILE${NC}"
    exit 1
fi

# Test SSH connection
echo -e "${BLUE}Testing connection to EC2 instance...${NC}"
ssh -i "$EC2_KEY" -o ConnectTimeout=10 "$EC2_INSTANCE" "echo Connection successful!" || {
    echo -e "${RED}Error: Could not connect to EC2 instance${NC}"
    exit 1
}

# Create remote directory if it doesn't exist
echo -e "${BLUE}Creating remote directory if needed...${NC}"
ssh -i "$EC2_KEY" "$EC2_INSTANCE" "mkdir -p $REMOTE_DIR"

# Transfer model file to EC2
echo -e "${BLUE}Transferring fixed CNN-LSTM model to EC2...${NC}"
scp -i "$EC2_KEY" "$LOCAL_MODEL_FILE" "$EC2_INSTANCE:$REMOTE_DIR/fixed_cnn_lstm_model.py"

# Make the script executable on the remote server
echo -e "${BLUE}Setting execution permissions...${NC}"
ssh -i "$EC2_KEY" "$EC2_INSTANCE" "chmod +x $REMOTE_DIR/fixed_cnn_lstm_model.py"

# Create data symlinks if needed
echo -e "${BLUE}Setting up data symlinks if needed...${NC}"
ssh -i "$EC2_KEY" "$EC2_INSTANCE" "
    # Create data directory
    mkdir -p $REMOTE_DIR/data
    
    # Create symlinks to actual data locations if they don't exist
    if [ ! -L $REMOTE_DIR/data/ravdess_features_cnn_fixed ]; then
        ln -s /home/ubuntu/emotion-recognition/data/ravdess_features_cnn_fixed $REMOTE_DIR/data/ravdess_features_cnn_fixed
    fi
    
    if [ ! -L $REMOTE_DIR/data/crema_d_features_cnn_fixed ]; then
        ln -s /home/ubuntu/emotion-recognition/data/crema_d_features_cnn_fixed $REMOTE_DIR/data/crema_d_features_cnn_fixed
    fi
    
    # Create models directory
    mkdir -p $REMOTE_DIR/models
"

# Start the training in a screen session
echo -e "${BLUE}Starting fixed CNN-LSTM training...${NC}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="fixed_cnn_lstm_training_${TIMESTAMP}.log"

ssh -i "$EC2_KEY" "$EC2_INSTANCE" "
    cd $REMOTE_DIR
    screen -dmS fixed_cnn_lstm bash -c \"
        echo 'Starting training at $(date)' > $LOG_FILE
        python3 fixed_cnn_lstm_model.py 2>&1 | tee -a $LOG_FILE
        echo 'Training completed at $(date)' >> $LOG_FILE
    \"
"

echo -e "${GREEN}Fixed CNN-LSTM model deployment complete!${NC}"
echo -e "${YELLOW}Training is running in a screen session on the EC2 instance.${NC}"
echo -e "Monitor training progress with: ${BLUE}ssh -i \"$HOME/Downloads/gpu-key.pem\" $EC2_INSTANCE \"tail -f $REMOTE_DIR/$LOG_FILE\"${NC}"
echo
echo -e "Log file: ${BLUE}$REMOTE_DIR/$LOG_FILE${NC}"
echo 
echo -e "${YELLOW}The fixed model resolves the dimension mismatch error by adding GlobalAveragePooling1D layer.${NC}"
