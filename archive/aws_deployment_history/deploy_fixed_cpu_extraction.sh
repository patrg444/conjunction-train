#!/bin/bash
# Script to deploy and run the fixed CPU-only CNN feature extraction
# This script uploads the fixed extraction script and runs it on the EC2 instance

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
echo -e "${YELLOW}     DEPLOY & RUN FIXED CPU-ONLY FEATURE EXTRACTION   ${NC}"
echo -e "${YELLOW}======================================================${NC}"

echo "Checking SSH connection..."
if ! ssh -i "$SSH_KEY" -o BatchMode=yes -o ConnectTimeout=5 "$SSH_HOST" "echo" &>/dev/null; then
    echo -e "${RED}❌ SSH connection failed. Check your SSH key and connection.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ SSH connection successful${NC}"
echo "Checking for any currently running extraction processes..."

RUNNING_PROCESSES=$(ssh -i "$SSH_KEY" "$SSH_HOST" "ps aux | grep preprocess_cnn_audio_features | grep -v grep | wc -l")
if [ "$RUNNING_PROCESSES" -gt 0 ]; then
    echo -e "${YELLOW}⚠️ Warning: $RUNNING_PROCESSES extraction processes are currently running.${NC}"
    echo "You may want to terminate them before starting a fresh extraction."
    echo -e "To kill them: ${YELLOW}ssh -i $SSH_KEY $SSH_HOST \"pkill -f 'preprocess_cnn_audio_features'\"${NC}"
    read -p "Do you want to kill the existing processes? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ssh -i "$SSH_KEY" "$SSH_HOST" "pkill -f 'preprocess_cnn_audio_features'"
        echo -e "${GREEN}✅ Existing processes terminated.${NC}"
    fi
fi

echo "Uploading fixed feature extraction script..."
scp -i "$SSH_KEY" fixed_preprocess_cnn_audio_features.py "$SSH_HOST:~/emotion-recognition/scripts/"

echo "Creating logs directory if needed..."
ssh -i "$SSH_KEY" "$SSH_HOST" "mkdir -p ~/emotion-recognition/logs"

echo "Creating output directories if needed..."
ssh -i "$SSH_KEY" "$SSH_HOST" "mkdir -p ~/emotion-recognition/data/ravdess_features_cnn_audio ~/emotion-recognition/data/crema_d_features_cnn_audio"

echo -e "${YELLOW}Ready to start extraction with the fixed CPU-only script.${NC}"
echo "This will run in a foreground terminal to show progress."
echo "The extraction will take 1-2 hours to complete."
read -p "Start extraction now? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Operation cancelled."
    exit 0
fi

# Run the extraction script with verbose output in the foreground
echo -e "${GREEN}Starting CPU-only feature extraction...${NC}"
echo "This will run in the foreground and show progress. Press Ctrl+C to interrupt."
echo -e "${YELLOW}=== Extraction Progress ====${NC}"

ssh -i "$SSH_KEY" "$SSH_HOST" "cd ~/emotion-recognition && PYTHONPATH=~/emotion-recognition python3 scripts/fixed_preprocess_cnn_audio_features.py --verbose --workers 4"

echo -e "${GREEN}=== Extraction Completed ====${NC}"
echo "To check the results:"
echo -e "  ${YELLOW}./verify_g5_feature_extraction_fix.sh${NC}"
echo -e "Once extraction is complete, restart training with:"
echo -e "  ${YELLOW}./run_audio_pooling_with_laughter.sh${NC}"
