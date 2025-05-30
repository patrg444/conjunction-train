#!/bin/bash
# Script to download the trained ATTN-CRNN v2 model from EC2

# Get EC2 IP
EC2_IP=$(cat aws_instance_ip.txt)
SSH_KEY="$HOME/Downloads/gpu-key.pem"  # Adjust if your key path is different

# Target directories
LOCAL_CHECKPOINT_DIR="checkpoints/attn_crnn_v2"
LOCAL_RESULTS_DIR="analysis_results/attn_crnn_v2"

# Create local directories if they don't exist
mkdir -p "${LOCAL_CHECKPOINT_DIR}"
mkdir -p "${LOCAL_RESULTS_DIR}"

echo "=== DOWNLOADING ATTN-CRNN V2 MODEL AND RESULTS ==="
echo "Source: EC2 instance ${EC2_IP}"
echo "Target: Local directories ${LOCAL_CHECKPOINT_DIR} and ${LOCAL_RESULTS_DIR}"

# Check if model files exist on the server
echo "Checking for model files on EC2..."
SSH_RESULT=$(ssh -i $SSH_KEY ubuntu@${EC2_IP} "cd ~/emotion_recognition && find checkpoints/attn_crnn_v2 -name \"*attn_crnn_model.keras\" 2>/dev/null | wc -l")

if [ "$SSH_RESULT" -eq "0" ]; then
    echo "No model files found on EC2 instance. Is training complete?"
    exit 1
fi

echo "Found $SSH_RESULT model files on EC2 instance."

# Check if results files exist on the server
echo "Checking for analysis results on EC2..."
RESULTS_COUNT=$(ssh -i $SSH_KEY ubuntu@${EC2_IP} "cd ~/emotion_recognition && find analysis_results/attn_crnn_v2 -type f 2>/dev/null | wc -l")

if [ "$RESULTS_COUNT" -eq "0" ]; then
    echo "No analysis results found on EC2 instance."
else
    echo "Found $RESULTS_COUNT analysis result files on EC2 instance."
fi

# Get the timestamp of the most recent model directory
TIMESTAMP=$(ssh -i $SSH_KEY ubuntu@${EC2_IP} "cd ~/emotion_recognition && ls -t checkpoints/attn_crnn_v2 | head -1")

if [ -z "$TIMESTAMP" ]; then
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    echo "Using current timestamp: $TIMESTAMP"
else
    echo "Found model checkpoint directory with timestamp: $TIMESTAMP"
fi

# Create timestamp-specific directories
LOCAL_TIMESTAMP_CHECKPOINT_DIR="${LOCAL_CHECKPOINT_DIR}/${TIMESTAMP}"
LOCAL_TIMESTAMP_RESULTS_DIR="${LOCAL_RESULTS_DIR}/${TIMESTAMP}"

mkdir -p "${LOCAL_TIMESTAMP_CHECKPOINT_DIR}"
mkdir -p "${LOCAL_TIMESTAMP_RESULTS_DIR}"

# Download the model files
echo "Downloading model files..."
scp -i $SSH_KEY ubuntu@${EC2_IP}:~/emotion_recognition/checkpoints/attn_crnn_v2/${TIMESTAMP}/*.keras \
    "${LOCAL_TIMESTAMP_CHECKPOINT_DIR}/"

# If that fails, try directly from the root directory
if [ $? -ne 0 ]; then
    echo "Trying to download model from the root directory..."
    scp -i $SSH_KEY ubuntu@${EC2_IP}:~/emotion_recognition/best_attn_crnn_model.keras \
        "${LOCAL_TIMESTAMP_CHECKPOINT_DIR}/"
fi

# Download the analysis results
echo "Downloading analysis results..."
scp -i $SSH_KEY ubuntu@${EC2_IP}:~/emotion_recognition/analysis_results/attn_crnn_v2/${TIMESTAMP}/* \
    "${LOCAL_TIMESTAMP_RESULTS_DIR}/" 2>/dev/null

# If that fails, try from the root analysis_results directory
if [ $? -ne 0 ]; then
    echo "Trying to download results from the root analysis directory..."
    scp -i $SSH_KEY ubuntu@${EC2_IP}:~/emotion_recognition/analysis_results/attn_crnn_v2/* \
        "${LOCAL_TIMESTAMP_RESULTS_DIR}/" 2>/dev/null
fi

# Download the log file
echo "Downloading training log..."
scp -i $SSH_KEY ubuntu@${EC2_IP}:~/emotion_recognition/training_attn_crnn_v2.log \
    "${LOCAL_TIMESTAMP_RESULTS_DIR}/" 2>/dev/null

# Create local links to the latest models for easy access
echo "Creating symbolic links to the latest model files..."
ln -sf "${LOCAL_TIMESTAMP_CHECKPOINT_DIR}/best_attn_crnn_model.keras" \
    "${LOCAL_CHECKPOINT_DIR}/best_attn_crnn_model.keras"

if [ -f "${LOCAL_TIMESTAMP_CHECKPOINT_DIR}/final_attn_crnn_model.keras" ]; then
    ln -sf "${LOCAL_TIMESTAMP_CHECKPOINT_DIR}/final_attn_crnn_model.keras" \
        "${LOCAL_CHECKPOINT_DIR}/final_attn_crnn_model.keras"
fi

# Check if we successfully downloaded the files
MODEL_COUNT=$(find "${LOCAL_TIMESTAMP_CHECKPOINT_DIR}" -name "*.keras" | wc -l)
RESULTS_COUNT=$(find "${LOCAL_TIMESTAMP_RESULTS_DIR}" -type f | wc -l)

echo
echo "Download complete."
echo "Downloaded $MODEL_COUNT model files to ${LOCAL_TIMESTAMP_CHECKPOINT_DIR}"
echo "Downloaded $RESULTS_COUNT result files to ${LOCAL_TIMESTAMP_RESULTS_DIR}"

if [ $MODEL_COUNT -eq 0 ]; then
    echo "WARNING: No model files were downloaded. Check if training has completed."
fi

# If there are any model files, verify them
if [ $MODEL_COUNT -gt 0 ]; then
    echo
    echo "Model files:"
    ls -lh "${LOCAL_TIMESTAMP_CHECKPOINT_DIR}"
    
    echo
    echo "Analysis results:"
    ls -lh "${LOCAL_TIMESTAMP_RESULTS_DIR}"
fi

echo
echo "The model can be loaded using:"
echo "  from tensorflow.keras.models import load_model"
echo "  model = load_model('${LOCAL_CHECKPOINT_DIR}/best_attn_crnn_model.keras')"
