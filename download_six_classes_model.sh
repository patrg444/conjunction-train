#!/bin/bash
# Download the trained wav2vec model with six continuous emotion classes

set -e  # Exit on error

# Define variables
EC2_HOST="ubuntu@54.162.134.77"
KEY_PATH="~/Downloads/gpu-key.pem"
REMOTE_DIR="/home/ubuntu/audio_emotion/checkpoints"
LOCAL_DIR="./checkpoints"
MODEL_NAME="wav2vec_six_classes"
TIMESTAMP=$(date +"%Y%m%d")

# Create local directory if it doesn't exist
mkdir -p $LOCAL_DIR

echo "Checking for available models on remote server..."

# Find the best model weights (supporting both naming patterns)
MODEL_PATH=$(ssh -i $KEY_PATH $EC2_HOST "ls -t $REMOTE_DIR/${MODEL_NAME}_best.weights.h5 $REMOTE_DIR/${MODEL_NAME}_*_best.weights.h5 2>/dev/null | head -n 1")

if [ -z "$MODEL_PATH" ]; then
    echo "No model weights found for ${MODEL_NAME}!"
    exit 1
fi

MODEL_FILENAME=$(basename "$MODEL_PATH")
echo "Found model: $MODEL_FILENAME"

# Download model weights
echo "Downloading model weights..."
scp -i $KEY_PATH $EC2_HOST:$MODEL_PATH $LOCAL_DIR/

# Download model history
HISTORY_PATH=${MODEL_PATH/_best.weights.h5/_history.json}
if ssh -i $KEY_PATH $EC2_HOST "test -f $HISTORY_PATH"; then
    echo "Downloading training history..."
    scp -i $KEY_PATH $EC2_HOST:$HISTORY_PATH $LOCAL_DIR/
fi

# Download validation summary
SUMMARY_PATH=${MODEL_PATH/_best.weights.h5/_validation_summary.csv}
if ssh -i $KEY_PATH $EC2_HOST "test -f $SUMMARY_PATH"; then
    echo "Downloading validation summary..."
    scp -i $KEY_PATH $EC2_HOST:$SUMMARY_PATH $LOCAL_DIR/
fi

# Download accuracy json
ACCURACY_PATH=${MODEL_PATH/_best.weights.h5/_validation_accuracy.json}
if ssh -i $KEY_PATH $EC2_HOST "test -f $ACCURACY_PATH"; then
    echo "Downloading validation accuracy..."
    scp -i $KEY_PATH $EC2_HOST:$ACCURACY_PATH $LOCAL_DIR/
fi

# Download training curves plot
CURVES_PATH=${MODEL_PATH/_best.weights.h5/_training_curves.png}
if ssh -i $KEY_PATH $EC2_HOST "test -f $CURVES_PATH"; then
    echo "Downloading training curves plot..."
    scp -i $KEY_PATH $EC2_HOST:$CURVES_PATH $LOCAL_DIR/
fi

echo "Download complete! Files saved to $LOCAL_DIR/"
echo ""
echo "Summary:"
if [ -f "$LOCAL_DIR/$(basename "$SUMMARY_PATH")" ]; then
    cat "$LOCAL_DIR/$(basename "$SUMMARY_PATH")"
fi
