#!/bin/bash
# Launch the 3D ResNet-LSTM video emotion model training on the combined RAVDESS and CREMA-D datasets
# This script:
# 1. Creates the required directories
# 2. Generates the manifest file with train/val/test splits
# 3. Launches the training inside a tmux session for persistence

set -e

# Configuration
MANIFEST_PATH="/home/ubuntu/datasets/video_manifest.csv"
OUTPUT_DIR="/home/ubuntu/emotion_full_video"
RAVDESS_DIR="/home/ubuntu/datasets/ravdess_videos"
CREMAD_DIR="/home/ubuntu/datasets/crema_d_videos"
BATCH_SIZE=8
EPOCHS=30
IMAGE_SIZE=112
FRAMES=48
WORKERS=4
SESSION_NAME="video_training"

# Check for GPU
if nvidia-smi &> /dev/null; then
    echo "GPU detected, enabling mixed precision training."
    FP16="--fp16"
else
    echo "No GPU detected, using CPU only (training will be slow)."
    FP16=""
fi

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Create the manifest file
echo "Generating manifest file with train/val/test splits..."
python3 scripts/generate_video_manifest.py \
    --ravdess_dir ${RAVDESS_DIR} \
    --cremad_dir ${CREMAD_DIR} \
    --output ${MANIFEST_PATH}

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "tmux is not installed. Installing..."
    sudo apt-get update && sudo apt-get install -y tmux
fi

# Kill existing session if it exists
tmux kill-session -t ${SESSION_NAME} 2>/dev/null || true

# Create new tmux session
echo "Creating tmux session: ${SESSION_NAME}"
tmux new-session -d -s ${SESSION_NAME}

# Launch training in tmux session
echo "Launching training... (use 'tmux attach -t ${SESSION_NAME}' to view)"
tmux send-keys -t ${SESSION_NAME} "cd $(pwd) && python3 scripts/train_video_full.py \
    --manifest_file ${MANIFEST_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --img_size ${IMAGE_SIZE} \
    --frames ${FRAMES} \
    --num_workers ${WORKERS} \
    ${FP16}" C-m

echo ""
echo "Training started in tmux session '${SESSION_NAME}'"
echo "To view the training progress:"
echo "  1. SSH into the server"
echo "  2. Run: tmux attach -t ${SESSION_NAME}"
echo "  3. Use Ctrl+B then D to detach without stopping training"
echo ""
echo "Training logs and models will be saved to: ${OUTPUT_DIR}"
echo "Monitor training with: ./monitor_video_training.sh"
