#!/bin/bash
set -e

# Set coloring for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

TRAINING_PID=14561
TRAINING_LOG_DIR="training_logs_humor/microsoft_deberta-v3-base_single"
CHECKPOINT_DIR="$TRAINING_LOG_DIR/checkpoints"
MAX_CHECK_COUNT=600  # Maximum number of checks (10 hours at 1 check per minute)
LOG_FILE="/tmp/deberta_training_log.txt"

# Function to stream training logs in real-time
stream_logs() {
    local temp_log="/tmp/temp_training_log.txt"
    # Initialize the temp log if it doesn't exist
    touch "$temp_log"

    # Watch the process output for log updates
    while ps -p $TRAINING_PID > /dev/null; do
        # Use tail -f to stream the log file continuously
        echo -e "${CYAN}Streaming training logs (Ctrl+C to stop streaming but continue monitoring)...${NC}"
        tail -f "$LOG_FILE" || true
        sleep 1
    done
}

# Create named pipe for logs if it doesn't exist
if [ ! -e "$LOG_FILE" ]; then
    touch "$LOG_FILE"
fi

echo -e "${BLUE}Starting DeBERTa v3 training monitor...${NC}"
echo -e "${BLUE}Monitoring training process with PID: ${YELLOW}$TRAINING_PID${NC}"
echo -e "${BLUE}Saving checkpoints to: ${YELLOW}$CHECKPOINT_DIR${NC}"
echo -e "${BLUE}Log file: ${YELLOW}$LOG_FILE${NC}"

# Attach to the process standard output if possible
if command -v strace &> /dev/null; then
    echo -e "${GREEN}Attaching to process output stream...${NC}"
    strace -p $TRAINING_PID -e write -s 1024 -f 2>&1 | grep -v "resumed>" | grep -oP "write\(1, \".*\"" | sed 's/write(1, "//g' | sed 's/".*//g' | sed 's/\\n/\n/g' > "$LOG_FILE" &
    STRACE_PID=$!
    echo -e "${GREEN}Started log streaming with PID $STRACE_PID${NC}"
    # Start streaming logs in a background process
    stream_logs &
    STREAM_PID=$!
fi

echo -e "${PURPLE}Real-time monitoring activated. Streaming logs and checking for checkpoints...${NC}"

count=0
while [ $count -lt $MAX_CHECK_COUNT ]; do
    # Check if training process is still running
    if ! ps -p $TRAINING_PID > /dev/null; then
        echo -e "${GREEN}Training process has completed!${NC}"
        # Kill the strace and streaming processes if they're running
        if [[ -n "$STRACE_PID" ]]; then
            kill $STRACE_PID 2>/dev/null || true
        fi
        if [[ -n "$STREAM_PID" ]]; then
            kill $STREAM_PID 2>/dev/null || true
        fi
        break
    fi
    
    # Check for .ckpt files in the checkpoint directories
    latest_ckpt=$(find "$TRAINING_LOG_DIR" -name "*.ckpt" | sort -V | tail -n 1)
    if [ -n "$latest_ckpt" ]; then
        ckpt_timestamp=$(stat -f "%Sm" "$latest_ckpt")
        echo -e "${GREEN}Latest checkpoint: ${YELLOW}$latest_ckpt${GREEN} (last modified: $ckpt_timestamp)${NC}"
    else
        echo -e "${YELLOW}No checkpoints found yet.${NC}"
    fi
    
    # Sleep for 60 seconds before checking again
    echo -e "${BLUE}Next checkpoint check in 60 seconds...${NC}"
    sleep 60
    ((count++))
done

# Create a summary of the training
echo -e "${GREEN}Creating training summary...${NC}"
SUMMARY_FILE="$TRAINING_LOG_DIR/training_summary.md"

cat << EOF > "$SUMMARY_FILE"
# DeBERTa v3 Humor Detection Model Training Summary

## Training Configuration

- **Model**: microsoft/deberta-v3-base
- **Task**: Binary classification (humor detection)
- **Training Dataset**: ur_funny_train_humor_cleaned.csv
- **Validation Dataset**: ur_funny_val_humor_cleaned.csv
- **Training Start Time**: $(date -r "$TRAINING_LOG_DIR")
- **Training End Time**: $(date)

## Hyperparameters

- **Max Sequence Length**: 128
- **Batch Size**: 16
- **Learning Rate**: 2.0e-05
- **Epochs**: 1
- **Optimizer**: AdamW
- **Weight Decay**: 0.01
- **Dropout**: 0.2
- **LR Scheduler**: Cosine
- **Gradient Clipping**: 1.0

## Files

EOF

find "$TRAINING_LOG_DIR" -type f | sort | while read -r file; do
    size=$(du -h "$file" | cut -f1)
    echo "- \`${file#$TRAINING_LOG_DIR/}\` ($size)" >> "$SUMMARY_FILE"
done

# Upload to EC2
echo -e "${GREEN}Uploading model to EC2...${NC}"
./upload_deberta_files.sh --include_manifests

echo -e "${GREEN}Done! DeBERTa v3 model has been trained and uploaded to EC2.${NC}"
