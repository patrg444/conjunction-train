#!/bin/bash
# Deployment script for G5 CNN feature extraction fix
# This script uploads the necessary files to fix the feature extraction issue on the G5 instance

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

SSH_KEY="$HOME/Downloads/gpu-key.pem"
SSH_USER="ubuntu"
AWS_IP="18.208.166.91"
SSH_HOST="$SSH_USER@$AWS_IP"

echo -e "${YELLOW}======================================================${NC}"
echo -e "${YELLOW}     G5 CNN FEATURE EXTRACTION FIX DEPLOYMENT      ${NC}"
echo -e "${YELLOW}======================================================${NC}"

echo "Checking SSH connection..."
if ! ssh -i "$SSH_KEY" -o BatchMode=yes -o ConnectTimeout=5 "$SSH_HOST" "echo" &>/dev/null; then
    echo -e "${RED}❌ SSH connection failed. Check your SSH key and connection.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ SSH connection successful${NC}"

# Create local temporary directory for files
TEMP_DIR=$(mktemp -d)
echo "Creating temporary directory: $TEMP_DIR"

# Copy files to temporary directory
echo "Preparing files for deployment..."
cp monitor_cnn_feature_extraction.sh $TEMP_DIR/
cp run_preprocess_cnn_features_fixed.sh $TEMP_DIR/
cp G5_CNN_FEATURE_EXTRACTION_FIX_README.md $TEMP_DIR/

# Make scripts executable
chmod +x $TEMP_DIR/*.sh

# Create tar file
TAR_FILE="$TEMP_DIR/g5_feature_extraction_fix.tar.gz"
tar -czf $TAR_FILE -C $TEMP_DIR .

echo "Uploading fix to EC2 instance..."
scp -i "$SSH_KEY" $TAR_FILE "$SSH_HOST:~/"

echo "Deploying fix on EC2 instance..."
ssh -i "$SSH_KEY" "$SSH_HOST" "
    echo 'Extracting files...'
    tar -xzf ~/g5_feature_extraction_fix.tar.gz -C ~/
    chmod +x ~/*.sh
    
    echo 'Checking existing processes...'
    EXTRACTION_PROCESSES=\$(ps aux | grep 'preprocess_cnn_audio_features.py' | grep -v grep | wc -l)
    
    if [ \$EXTRACTION_PROCESSES -eq 0 ]; then
        echo 'No extraction processes found. Starting extraction...'
        cd ~ && ./run_preprocess_cnn_features_fixed.sh
    else
        echo 'Extraction processes already running. Use monitor script to check progress.'
    fi
    
    echo 'Cleaning up...'
    rm ~/g5_feature_extraction_fix.tar.gz
"

# Cleanup temporary directory
rm -rf $TEMP_DIR

echo -e "${GREEN}✅ Feature extraction fix deployed successfully${NC}"
echo ""
echo "To monitor the extraction progress:"
echo -e "  ${YELLOW}ssh -i $SSH_KEY $SSH_HOST \"./monitor_cnn_feature_extraction.sh\"${NC}"
echo ""
echo "The extraction process will take 1-2 hours. Once complete, restart training with:"
echo -e "  ${YELLOW}ssh -i $SSH_KEY $SSH_HOST \"cd ~/emotion-recognition && ./run_audio_pooling_with_laughter.sh\"${NC}"
echo -e "${YELLOW}======================================================${NC}"
