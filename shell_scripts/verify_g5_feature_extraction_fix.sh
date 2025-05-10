#!/bin/bash
# Script to verify that the G5 CNN audio feature extraction fix has been applied successfully

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SSH_KEY="$HOME/Downloads/gpu-key.pem"
SSH_USER="ubuntu"
AWS_IP="18.208.166.91"
SSH_HOST="$SSH_USER@$AWS_IP"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}   G5 FEATURE EXTRACTION FIX VERIFICATION - ${TIMESTAMP}${NC}"
echo -e "${BLUE}======================================================${NC}"

# Check SSH connection
echo -e "\n${YELLOW}[1/6]${NC} Testing SSH connection..."
if ! ssh -i "$SSH_KEY" -o BatchMode=yes -o ConnectTimeout=5 "$SSH_HOST" "echo SSH connection successful" &>/dev/null; then
    echo -e "${RED}❌ SSH connection failed. Check your SSH key and connection.${NC}"
    exit 1
else
    ssh -i "$SSH_KEY" -o BatchMode=yes -o ConnectTimeout=5 "$SSH_HOST" "echo SSH connection successful"
    echo -e "${GREEN}✅ SSH connection successful.${NC}"
fi

# Check if CNN audio features were extracted
echo -e "\n${YELLOW}[2/6]${NC} Verifying CNN audio features were extracted..."
CNN_FEATURES=$(ssh -i "$SSH_KEY" "$SSH_HOST" "
    echo '- CNN Audio Features Directories:'
    du -sh ~/emotion-recognition/data/ravdess_features_cnn_audio/ ~/emotion-recognition/data/crema_d_features_cnn_audio/
    
    echo '- RAVDESS CNN Features:' 
    find ~/emotion-recognition/data/ravdess_features_cnn_audio -name '*.npy' 2>/dev/null | wc -l
    
    echo '- CREMA-D CNN Features:'
    find ~/emotion-recognition/data/crema_d_features_cnn_audio -name '*.npy' 2>/dev/null | wc -l
")
echo "$CNN_FEATURES"

# Extract file counts
RAVDESS_COUNT=$(echo "$CNN_FEATURES" | grep -A 1 'RAVDESS CNN Features' | tail -1)
CREMAD_COUNT=$(echo "$CNN_FEATURES" | grep -A 1 'CREMA-D CNN Features' | tail -1)

# Check if features were extracted successfully
if [ "$RAVDESS_COUNT" -eq 0 ] || [ "$CREMAD_COUNT" -eq 0 ]; then
    echo -e "${RED}❌ One or both datasets have no extracted features.${NC}"
else
    # Check if the expected number of files were extracted (approximately)
    if [ "$RAVDESS_COUNT" -lt 1400 ] || [ "$CREMAD_COUNT" -lt 7400 ]; then
        echo -e "${YELLOW}⚠️ Warning: The number of extracted features seems low.${NC}"
        echo -e "Expected: ~1440 RAVDESS, ~7442 CREMA-D"
        echo -e "Actual: $RAVDESS_COUNT RAVDESS, $CREMAD_COUNT CREMA-D"
    else
        echo -e "${GREEN}✅ CNN features extracted successfully.${NC}"
        echo -e "RAVDESS: $RAVDESS_COUNT/1440 files"
        echo -e "CREMA-D: $CREMAD_COUNT/7442 files"
    fi
fi

# Check GPU utilization
echo -e "\n${YELLOW}[3/6]${NC} Checking GPU utilization..."
GPU_STATS=$(ssh -i "$SSH_KEY" "$SSH_HOST" "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits")
echo -e "- Current GPU Statistics:"
echo "$GPU_STATS"

# Extract GPU utilization
GPU_UTIL=$(echo "$GPU_STATS" | awk -F ', ' '{print $1}')
if [ "$GPU_UTIL" -lt 25 ]; then
    echo -e "${YELLOW}⚠️ Warning: GPU utilization is still low (${GPU_UTIL}%). Training might not be using GPU effectively.${NC}"
else
    echo -e "${GREEN}✅ GPU utilization is adequate (${GPU_UTIL}%).${NC}"
fi

# Check training logs for evidence of real data processing
echo -e "\n${YELLOW}[4/6]${NC} Checking training logs for evidence of real data processing..."
LATEST_LOG=$(ssh -i "$SSH_KEY" "$SSH_HOST" "
    echo '- Recent log entries:'
    LATEST_TRAIN_LOG=\$(ls -t ~/emotion-recognition/logs/training_*.log 2>/dev/null | head -1)
    if [ -n \"\$LATEST_TRAIN_LOG\" ]; then
        echo \"- Latest training log: \$LATEST_TRAIN_LOG\"
        SAMPLE_COUNT=\$(grep -o 'samples: [0-9]*' \$LATEST_TRAIN_LOG | tail -1)
        echo \"\$SAMPLE_COUNT\"
    fi
    
    echo -e '\n- Recent epoch times:'
    if [ -n \"\$LATEST_TRAIN_LOG\" ]; then
        grep 'Epoch [0-9]*/[0-9]*' \$LATEST_TRAIN_LOG | tail -3
    fi
")
echo "$LATEST_LOG"

# Extract sample count
SAMPLE_COUNT=$(echo "$LATEST_LOG" | grep -o 'samples: [0-9]*' | tail -1 | awk '{print $2}')
if [ -z "$SAMPLE_COUNT" ] || [ "$SAMPLE_COUNT" -lt 5000 ]; then
    echo -e "${YELLOW}⚠️ Warning: Training sample count looks suspicious (${SAMPLE_COUNT}).${NC}"
fi

# Check if training process is running
echo -e "\n${YELLOW}[5/6]${NC} Checking if training process is running..."
TRAINING_PROCESSES=$(ssh -i "$SSH_KEY" "$SSH_HOST" "
    TRAIN_PROCESS=\$(pgrep -fa 'train_audio_pooling.*\.py');
    if [ -z \"\$TRAIN_PROCESS\" ]; then
        echo '❌ No training process appears to be running.'
    else
        echo '✅ Training process is active:'
        echo \"\$TRAIN_PROCESS\"
        
        # Get process resource usage
        echo -e '\n- Training process resource usage:'
        ps -p \$(echo \$TRAIN_PROCESS | awk '{print \$1}') -o pid,ppid,%cpu,%mem,etime,cmd | head -1
        ps -p \$(echo \$TRAIN_PROCESS | awk '{print \$1}') -o pid,ppid,%cpu,%mem,etime,cmd | grep -v PID
    fi
")
echo "$TRAINING_PROCESSES"

# Overall assessment
echo -e "\n${YELLOW}[6/6]${NC} Overall assessment..."
echo -e "Based on the checks above, here's an assessment of the fix:\n"

# Were files extracted?
if [ "$RAVDESS_COUNT" -eq 0 ] || [ "$CREMAD_COUNT" -eq 0 ]; then
    echo -e "${RED}❌ Feature extraction has not yet completed${NC}"
    
    # Check if extraction is still running
    EXTRACT_PROCESSES=$(ssh -i "$SSH_KEY" "$SSH_HOST" "ps aux | grep 'preprocess_cnn_audio_features.py' | grep -v grep | wc -l")
    if [ "$EXTRACT_PROCESSES" -gt 0 ]; then
        echo -e "${YELLOW}Feature extraction is still in progress. Check back later.${NC}"
        echo -e "To monitor progress: ./monitor_cnn_feature_extraction.sh"
    else
        echo -e "${RED}Feature extraction has stopped but did not complete successfully!${NC}"
        echo -e "Consider restarting extraction with: ./run_preprocess_cnn_features_fixed.sh"
    fi
else
    echo -e "${GREEN}✅ Feature extraction has completed successfully!${NC}"
    
    # Is training running?
    if echo "$TRAINING_PROCESSES" | grep -q "No training process appears to be running"; then
        echo -e "${YELLOW}Training process is not running. Start it with:${NC}"
        echo -e "./run_audio_pooling_with_laughter.sh"
    else
        echo -e "${GREEN}✅ Training process is running. Monitor progress with:${NC}"
        echo -e "./monitor_training_$(date +%Y%m%d)_*.sh"
    fi
fi

echo -e "\nThis is a point-in-time verification. For ongoing monitoring use:"
echo -e "./enhanced_monitor_g5.sh"
echo -e "\nFor detailed TensorBoard visualization use:"
echo -e "./setup_tensorboard_tunnel.sh"
echo -e "Then open http://localhost:6006 in your browser"
echo -e "\n${BLUE}======================================================${NC}"
echo -e "Verification complete! Run periodically to monitor progress."
echo -e "${BLUE}======================================================${NC}"
