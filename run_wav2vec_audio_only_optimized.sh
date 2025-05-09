#!/bin/bash
# Script to run audio-only wav2vec emotion recognition training on EC2
# Optimized version with hyper-parameter tuning and mixed precision

echo "Starting optimized audio-only wav2vec emotion recognition training..."

# Parameters
EC2_USER="ubuntu"
EC2_IP="54.162.134.77"  # Update with your EC2 instance IP
SSH_KEY="~/Downloads/gpu-key.pem"  # Update with your SSH key path
REMOTE_DIR="/home/ubuntu/audio_emotion"

# Optimization tweaks
# - Increased epochs from 100 to 200 (early stopping will prevent overfitting)
# - Lowered learning rate from 6e-4 to 3e-4 for better convergence
# - Increased batch size from 64 to 128 for faster training
# - Mixed precision is enabled in the script itself

# Create necessary directories
ssh -i $SSH_KEY $EC2_USER@$EC2_IP "mkdir -p $REMOTE_DIR/scripts"

# Upload both the training script and data preparation script
echo "Uploading scripts to EC2..."
scp -i $SSH_KEY scripts/train_wav2vec_audio_only.py $EC2_USER@$EC2_IP:$REMOTE_DIR/scripts/
scp -i $SSH_KEY scripts/prepare_wav2vec_data.py $EC2_USER@$EC2_IP:$REMOTE_DIR/scripts/

# Ensure data directory structure is ready
echo "Running data preparation script..."
ssh -i $SSH_KEY $EC2_USER@$EC2_IP "cd $REMOTE_DIR && python scripts/prepare_wav2vec_data.py"

# Create directories for logs and checkpoints
ssh -i $SSH_KEY $EC2_USER@$EC2_IP "mkdir -p $REMOTE_DIR/logs $REMOTE_DIR/checkpoints"

# Stop any existing training processes
echo "Stopping any existing training processes..."
ssh -i $SSH_KEY $EC2_USER@$EC2_IP "pkill -f train_wav2vec_audio_only || true"

# Run the audio-only training script with the optimized parameters
echo "Starting optimized audio-only training..."
ssh -i $SSH_KEY $EC2_USER@$EC2_IP "cd $REMOTE_DIR && nohup python scripts/train_wav2vec_audio_only.py \
    --features_dir data \
    --mean_path embedding_mean.npy \
    --std_path embedding_std.npy \
    --epochs 200 \
    --batch_size 128 \
    --lr 3e-4 \
    --log_dir logs \
    --checkpoint_dir checkpoints \
    --seed 42 \
    > train_wav2vec_audio_only.log 2>&1 &"

echo "Optimized audio-only training started!"
echo "Monitor training with:"
echo "./monitor_wav2vec_training.sh $SSH_KEY $EC2_IP"
echo ""
echo "Set up TensorBoard with:"
echo "./setup_tensorboard_tunnel.sh $SSH_KEY $EC2_IP"
echo ""
echo "When training completes, download the model with:"
echo "./download_wav2vec_model.sh $SSH_KEY $EC2_IP"
