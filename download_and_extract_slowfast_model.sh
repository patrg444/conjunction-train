#!/usr/bin/env bash
# Download and extract the SlowFast model weights for deployment

# Set variables
EC2_INSTANCE="ubuntu@54.162.134.77"
KEY_PATH="$HOME/Downloads/gpu-key.pem"
REMOTE_MODEL_PATH="/home/ubuntu/emotion_slowfast/slowfast_emotion_20250422_040528_best.pt"
LOCAL_FULL_MODEL="slowfast_emotion_full_checkpoint.pt"
LOCAL_EXTRACTED_MODEL="slowfast_emotion_video_only_92.9.pt"

# Create the models directory if it doesn't exist
mkdir -p models

echo "=== SlowFast Model Deployment Utility ==="
echo "1. Downloading full model checkpoint from EC2..."
scp -i "$KEY_PATH" "$EC2_INSTANCE:$REMOTE_MODEL_PATH" "models/$LOCAL_FULL_MODEL"

if [ $? -ne 0 ]; then
    echo "Error: Failed to download the model from EC2"
    exit 1
fi

echo "2. Extracting model weights using extract_model_weights.py..."
python extract_model_weights.py --input "models/$LOCAL_FULL_MODEL" --output "models/$LOCAL_EXTRACTED_MODEL"

if [ $? -ne 0 ]; then
    echo "Error: Failed to extract model weights"
    exit 1
fi

echo "=== Model Export Complete ==="
echo "Full checkpoint: models/$LOCAL_FULL_MODEL"
echo "Deployment model: models/$LOCAL_EXTRACTED_MODEL"
echo
echo "To use this model in your application, load it with:"
echo "model = YourModelClass(...)"
echo "model.load_state_dict(torch.load('models/$LOCAL_EXTRACTED_MODEL'))"
