#!/bin/bash
# Download trained wav2vec fixed model from the EC2 instance
# This script downloads the model weights, training history, and logs

# Define colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "================================================"
echo -e "${BLUE}Download Wav2Vec Fixed Model${NC}"
echo "================================================"

# Ensure the gpu key has the right permissions
chmod 400 ~/Downloads/gpu-key.pem

# Create local directory for downloads
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DOWNLOAD_DIR="wav2vec_model_${TIMESTAMP}"
mkdir -p "$DOWNLOAD_DIR"
echo -e "Downloading to ${YELLOW}${DOWNLOAD_DIR}${NC} directory"

# Find the best model on the server
echo -e "\n${BLUE}Finding the best model...${NC}"
BEST_MODEL=$(ssh -i ~/Downloads/gpu-key.pem ubuntu@54.162.134.77 \
    "cd /home/ubuntu/audio_emotion && ls -t checkpoints/wav2vec_audio_only_fixed_*_best.weights.h5 2>/dev/null | head -1")

if [ -z "$BEST_MODEL" ]; then
    echo -e "${RED}No best model weights found!${NC}"
    echo "Check if training completed successfully or if the model was saved with a different name."
    echo "Looking for final model instead..."
    
    FINAL_MODEL=$(ssh -i ~/Downloads/gpu-key.pem ubuntu@54.162.134.77 \
        "cd /home/ubuntu/audio_emotion && ls -t checkpoints/wav2vec_audio_only_fixed_*_final.weights.h5 2>/dev/null | head -1")
    
    if [ -z "$FINAL_MODEL" ]; then
        echo -e "${RED}No final model weights found either!${NC}"
        echo "Check if training is still in progress or failed."
        
        # Offer to download partial checkpoints in case user still wants them
        read -p "Would you like to download any available checkpoint? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            ANY_MODEL=$(ssh -i ~/Downloads/gpu-key.pem ubuntu@54.162.134.77 \
                "cd /home/ubuntu/audio_emotion && ls -t checkpoints/wav2vec_audio_only_fixed_*.weights.h5 2>/dev/null | head -1")
            if [ -z "$ANY_MODEL" ]; then
                echo -e "${RED}No checkpoint files found at all.${NC}"
                exit 1
            else
                BEST_MODEL=$ANY_MODEL
                echo -e "Using most recent checkpoint: ${YELLOW}$BEST_MODEL${NC}"
            fi
        else
            exit 1
        fi
    else
        BEST_MODEL=$FINAL_MODEL
        echo -e "Using final model: ${YELLOW}$BEST_MODEL${NC}"
    fi
fi

# Extract the model timestamp from the filename for finding related files
MODEL_BASENAME=$(basename "$BEST_MODEL")
MODEL_TIMESTAMP=$(echo "$MODEL_BASENAME" | grep -o "[0-9]\{8\}_[0-9]\{6\}")

if [ -z "$MODEL_TIMESTAMP" ]; then
    echo -e "${RED}Couldn't extract timestamp from model filename!${NC}"
    echo "Using generic patterns to find related files."
    HISTORY_PATTERN="*history.json"
    LOG_PATTERN="wav2vec_fixed_training_*.log"
else
    echo -e "Model timestamp: ${GREEN}$MODEL_TIMESTAMP${NC}"
    HISTORY_PATTERN="*${MODEL_TIMESTAMP}*history.json"
    LOG_PATTERN="wav2vec_fixed_training_${MODEL_TIMESTAMP}*.log"
fi

# Download best model weights
echo -e "\n${BLUE}Downloading model weights...${NC}"
scp -i ~/Downloads/gpu-key.pem "ubuntu@54.162.134.77:/home/ubuntu/audio_emotion/$BEST_MODEL" "$DOWNLOAD_DIR/"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Model weights downloaded successfully!${NC}"
else
    echo -e "${RED}Failed to download model weights!${NC}"
    exit 1
fi

# Download training history
echo -e "\n${BLUE}Downloading training history...${NC}"
HISTORY_FILE=$(ssh -i ~/Downloads/gpu-key.pem ubuntu@54.162.134.77 \
    "cd /home/ubuntu/audio_emotion && ls -t checkpoints/$HISTORY_PATTERN 2>/dev/null | head -1")
    
if [ ! -z "$HISTORY_FILE" ]; then
    scp -i ~/Downloads/gpu-key.pem "ubuntu@54.162.134.77:/home/ubuntu/audio_emotion/$HISTORY_FILE" "$DOWNLOAD_DIR/"
    echo -e "${GREEN}Training history downloaded!${NC}"
else
    echo -e "${YELLOW}Warning: No training history found matching pattern $HISTORY_PATTERN${NC}"
fi

# Download log file
echo -e "\n${BLUE}Downloading log file...${NC}"
LOG_FILE=$(ssh -i ~/Downloads/gpu-key.pem ubuntu@54.162.134.77 \
    "cd /home/ubuntu/audio_emotion && ls -t $LOG_PATTERN 2>/dev/null | head -1")
    
if [ ! -z "$LOG_FILE" ]; then
    scp -i ~/Downloads/gpu-key.pem "ubuntu@54.162.134.77:/home/ubuntu/audio_emotion/$LOG_FILE" "$DOWNLOAD_DIR/"
    echo -e "${GREEN}Log file downloaded!${NC}"
else
    echo -e "${YELLOW}Warning: No log file found matching pattern $LOG_PATTERN${NC}"
fi

# Download TensorBoard logs if available
echo -e "\n${BLUE}Downloading TensorBoard logs...${NC}"
TB_DIR=$(ssh -i ~/Downloads/gpu-key.pem ubuntu@54.162.134.77 \
    "cd /home/ubuntu/audio_emotion/logs && ls -d wav2vec_audio_only_fixed_$MODEL_TIMESTAMP 2>/dev/null")
    
if [ ! -z "$TB_DIR" ]; then
    mkdir -p "$DOWNLOAD_DIR/logs"
    scp -r -i ~/Downloads/gpu-key.pem "ubuntu@54.162.134.77:/home/ubuntu/audio_emotion/logs/$TB_DIR" "$DOWNLOAD_DIR/logs/"
    echo -e "${GREEN}TensorBoard logs downloaded!${NC}"
else
    echo -e "${YELLOW}Warning: No TensorBoard logs found matching timestamp $MODEL_TIMESTAMP${NC}"
fi

# Check the download directory
echo -e "\n${BLUE}Downloaded files:${NC}"
ls -la "$DOWNLOAD_DIR"

# Generate training curve if history file exists
DOWNLOADED_HISTORY=$(find "$DOWNLOAD_DIR" -name "*history.json" | head -1)
if [ ! -z "$DOWNLOADED_HISTORY" ]; then
    echo -e "\n${BLUE}Generating training curve...${NC}"
    if [ -f "scripts/plot_training_curve.py" ]; then
        python3 scripts/plot_training_curve.py --history_file "$DOWNLOADED_HISTORY" --metric both --output_file "$DOWNLOAD_DIR/training_curve.png"
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}Training curve generated: $DOWNLOAD_DIR/training_curve.png${NC}"
        else
            echo -e "${RED}Failed to generate training curve.${NC}"
        fi
    else
        echo -e "${YELLOW}Warning: plot_training_curve.py not found, skipping curve generation${NC}"
    fi
fi

echo -e "\n${GREEN}Download complete!${NC}"
echo -e "Files downloaded to: ${YELLOW}${DOWNLOAD_DIR}${NC}"
