#!/bin/bash
# Deploy the fixed wav2vec training scripts (v2) to EC2 and run them
# This version includes the fix for the optimizer configuration (clipnorm/clipvalue conflict)

# Ensure the gpu key has the right permissions
chmod 400 ~/Downloads/gpu-key.pem

echo "Deploying fixed wav2vec training scripts (v2) to EC2..."

# Upload all the necessary files
echo "Uploading scripts..."
scp -i ~/Downloads/gpu-key.pem \
  scripts/wav2vec_fixed_loader.py \
  scripts/train_wav2vec_audio_only_fixed_v2.py \
  run_wav2vec_audio_only_fixed_v2.sh \
  ubuntu@54.162.134.77:/home/ubuntu/audio_emotion/scripts/

# Make the script executable on the remote server
echo "Setting up permissions..."
ssh -i ~/Downloads/gpu-key.pem ubuntu@54.162.134.77 "chmod +x /home/ubuntu/audio_emotion/scripts/run_wav2vec_audio_only_fixed_v2.sh"

# Create proper symlink in main directory for ease of use
echo "Creating symlinks..."
ssh -i ~/Downloads/gpu-key.pem ubuntu@54.162.134.77 "cd /home/ubuntu/audio_emotion && ln -sf scripts/run_wav2vec_audio_only_fixed_v2.sh run_wav2vec_audio_only_fixed_v2.sh"

# Edit the run script to make sure it has proper paths
echo "Adjusting script paths..."
ssh -i ~/Downloads/gpu-key.pem ubuntu@54.162.134.77 "cd /home/ubuntu/audio_emotion && sed -i 's|python -m scripts.train_wav2vec_audio_only_fixed_v2|python -m scripts.train_wav2vec_audio_only_fixed_v2|g' run_wav2vec_audio_only_fixed_v2.sh"

echo "Setting up TensorBoard..."
# Set up TensorBoard monitoring in the background, so we have logs ready
ssh -i ~/Downloads/gpu-key.pem ubuntu@54.162.134.77 "cd /home/ubuntu/audio_emotion && mkdir -p logs && nohup tensorboard --logdir=logs --port=6006 --host=0.0.0.0 > tensorboard.log 2>&1 &"

echo "Starting training with improved numerical stability measures (v2)..."
# Start the training in a screen session so it continues after disconnection
ssh -i ~/Downloads/gpu-key.pem ubuntu@54.162.134.77 "cd /home/ubuntu/audio_emotion && screen -dmS wav2vec_training_v2 bash -c './run_wav2vec_audio_only_fixed_v2.sh; exec bash'"

echo "Training script has been launched!"
echo "Run this command to set up TensorBoard monitoring:"
echo "ssh -i ~/Downloads/gpu-key.pem -L 6006:localhost:6006 ubuntu@54.162.134.77"
echo ""
echo "Then open http://localhost:6006 in your browser."
echo ""
echo "To check on the training progress, run:"
echo "ssh -i ~/Downloads/gpu-key.pem ubuntu@54.162.134.77 \"tail -f /home/ubuntu/audio_emotion/wav2vec_fixed_training_v2_*.log\""
