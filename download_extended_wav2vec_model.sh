#!/bin/bash
# Script to download the best weights from extended wav2vec training

# Connection details
EC2_HOST="ubuntu@54.162.134.77"
KEY_PATH="~/Downloads/gpu-key.pem"
REMOTE_DIR="/home/ubuntu/audio_emotion/checkpoints"

# Create local directory for downloaded models
LOCAL_DIR="extended_models"
mkdir -p $LOCAL_DIR

# Find the latest extended model (the most recently created one)
echo "Finding the latest extended model checkpoint..."
LATEST_MODEL=$(ssh -i $KEY_PATH $EC2_HOST "ls -t $REMOTE_DIR/wav2vec_continued_extended_*_best.weights.h5 2>/dev/null | head -1")

if [ -z "$LATEST_MODEL" ]; then
    echo "No extended model checkpoints found. Has the training completed?"
    echo "Looking for any extended model checkpoints..."
    
    # Try a broader search pattern
    LATEST_MODEL=$(ssh -i $KEY_PATH $EC2_HOST "ls -t $REMOTE_DIR/*extended*_best.weights.h5 2>/dev/null | head -1")
    
    if [ -z "$LATEST_MODEL" ]; then
        echo "Still no extended model checkpoints found. Training may still be in progress."
        exit 1
    fi
fi

echo "Found model: $LATEST_MODEL"

# Get model name without path
MODEL_FILENAME=$(basename "$LATEST_MODEL")

# Download the model weights
echo "Downloading model weights..."
scp -i $KEY_PATH $EC2_HOST:$LATEST_MODEL $LOCAL_DIR/

# Download the validation summary if available
VAL_SUMMARY=${LATEST_MODEL/_best.weights.h5/_validation_summary.csv}
echo "Looking for validation summary at $VAL_SUMMARY"
ssh -i $KEY_PATH $EC2_HOST "test -f $VAL_SUMMARY" && scp -i $KEY_PATH $EC2_HOST:$VAL_SUMMARY $LOCAL_DIR/

# Download the validation accuracy JSON if available
VAL_JSON=${LATEST_MODEL/_best.weights.h5/_validation_accuracy.json}
echo "Looking for validation accuracy JSON at $VAL_JSON"
ssh -i $KEY_PATH $EC2_HOST "test -f $VAL_JSON" && scp -i $KEY_PATH $EC2_HOST:$VAL_JSON $LOCAL_DIR/

# Get the validation accuracy from the JSON file if it exists locally
LOCAL_VAL_JSON="$LOCAL_DIR/$(basename "$VAL_JSON")"
if [ -f "$LOCAL_VAL_JSON" ]; then
    ACCURACY=$(grep -o '"val_accuracy": [0-9.]*' "$LOCAL_VAL_JSON" | cut -d' ' -f2)
    echo "Model validation accuracy: $ACCURACY"
fi

echo "Model and related files have been downloaded to $LOCAL_DIR/"
echo "Model weights file: $LOCAL_DIR/$MODEL_FILENAME"
