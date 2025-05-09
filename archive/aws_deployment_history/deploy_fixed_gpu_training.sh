#!/usr/bin/env bash
# Deploy and run the fixed audio-pooling LSTM training on AWS G5 GPU

SSH_KEY="$HOME/Downloads/gpu-key.pem"
SSH_USER="ubuntu"
AWS_IP="18.208.166.91"
SSH_HOST="$SSH_USER@$AWS_IP"

EPOCHS=100      # Full training run
BATCH_SIZE=256  # Large batch size for GPU
SEQ_LEN=45      # Default sequence length (15 FPS × 3 s window) for compatibility

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REMOTE_SCRIPT="train_g5_fixed_${TIMESTAMP}.sh"
REMOTE_LOG="train_g5_fixed_${TIMESTAMP}.log"
DOWNLOAD_SCRIPT="download_g5_fixed_model_${TIMESTAMP}.sh"

echo "Creating training script for AWS GPU instance..."

# Create the remote training script
cat > /tmp/remote_train.sh << 'EOL'
#!/usr/bin/env bash
# G5 GPU Training script for fixed audio pooling LSTM
set -e

cd ~/emotion-recognition

# Verify CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "CUDA not available! Please use a GPU instance."
    exit 1
fi

# Install dependencies if needed
echo "Installing requirements..."
pip install -r requirements.txt

# Setup directories and data
mkdir -p models/dynamic_padding_no_leakage

# Check data presence and size
RAVDESS_DIR="ravdess_features_facenet"
CREMA_DIR="crema_d_features_facenet"
RAVDESS_SIZE=$(du -s ${RAVDESS_DIR} 2>/dev/null | awk '{print $1}' || echo 0)
CREMA_SIZE=$(du -s ${CREMA_DIR} 2>/dev/null | awk '{print $1}' || echo 0)

if [[ ${RAVDESS_SIZE} -lt 1600000 || ${CREMA_SIZE} -lt 1000000 ]]; then
    echo "Warning: Data size smaller than expected or not present"
    echo "RAVDESS: ${RAVDESS_SIZE} bytes, CREMA-D: ${CREMA_SIZE} bytes"
fi

# Copy normalization files to their expected location
if [[ -f models/audio_normalization_stats.pkl ]]; then
    cp models/audio_normalization_stats.pkl models/dynamic_padding_no_leakage/
fi
if [[ -f models/video_normalization_stats.pkl ]]; then
    cp models/video_normalization_stats.pkl models/dynamic_padding_no_leakage/
fi

# Start training
echo "Starting training with batch size BATCH_SIZE epochs EPOCHS..."
export CUDA_VISIBLE_DEVICES=0
python scripts/train_audio_pooling_lstm_fixed.py \
    --epochs EPOCHS \
    --batch_size BATCH_SIZE \
    --seq_len SEQ_LEN \
    --gpu 0 \
    > ~/train_output.log 2>&1

echo "Training completed. Model saved to models/dynamic_padding_no_leakage/model_best.h5"
EOL

# Replace placeholders with actual values
sed -i "s/BATCH_SIZE/$BATCH_SIZE/g" /tmp/remote_train.sh
sed -i "s/EPOCHS/$EPOCHS/g" /tmp/remote_train.sh
sed -i "s/SEQ_LEN/$SEQ_LEN/g" /tmp/remote_train.sh

# Create download script for retrieving model after training
cat > ${DOWNLOAD_SCRIPT} << EOL
#!/usr/bin/env bash
# Download model from GPU instance
SSH_KEY="$HOME/Downloads/gpu-key.pem"
SSH_USER="ubuntu"
AWS_IP="$AWS_IP"
SSH_HOST="\$SSH_USER@\$AWS_IP"

echo "Checking if training is complete..."
TRAINING_STATUS=\$(ssh -i "\$SSH_KEY" "\$SSH_HOST" "grep -c 'Training completed' ~/train_output.log || echo 0")

if [[ \$TRAINING_STATUS -eq 0 ]]; then
    echo "Training not yet complete. Viewing current status:"
    ssh -i "\$SSH_KEY" "\$SSH_HOST" "tail -n 30 ~/train_output.log"
    exit 1
fi

echo "Downloading model from GPU instance..."
MODEL_DIR="models/g5_fixed_${TIMESTAMP}"
mkdir -p \$MODEL_DIR

# Copy model files
scp -i "\$SSH_KEY" "\$SSH_HOST:~/emotion-recognition/models/dynamic_padding_no_leakage/model_best.h5" "\$MODEL_DIR/"
scp -i "\$SSH_KEY" "\$SSH_HOST:~/emotion-recognition/models/dynamic_padding_no_leakage/model_info.json" "\$MODEL_DIR/"
scp -i "\$SSH_KEY" "\$SSH_HOST:~/emotion-recognition/models/dynamic_padding_no_leakage/audio_normalization_stats.pkl" "\$MODEL_DIR/"
scp -i "\$SSH_KEY" "\$SSH_HOST:~/emotion-recognition/models/dynamic_padding_no_leakage/video_normalization_stats.pkl" "\$MODEL_DIR/"
scp -i "\$SSH_KEY" "\$SSH_HOST:~/train_output.log" "\$MODEL_DIR/training.log"

echo "Model downloaded to \$MODEL_DIR"
EOL

chmod +x ${DOWNLOAD_SCRIPT}

# Upload script to EC2 instance
echo "Uploading training script to EC2 instance..."
scp -i "$SSH_KEY" /tmp/remote_train.sh "$SSH_HOST":~/${REMOTE_SCRIPT}

# Make the script executable and run it
echo "Making script executable and launching training in the background..."
ssh -i "$SSH_KEY" "$SSH_HOST" "chmod +x ~/${REMOTE_SCRIPT} && nohup ~/${REMOTE_SCRIPT} > ~/${REMOTE_LOG} 2>&1 &"

echo "Training job has been started on the EC2 instance."
echo "Monitor progress with: ssh -i $SSH_KEY $SSH_HOST tail -f ~/${REMOTE_LOG}"
echo "After training completes, download the model using: ./${DOWNLOAD_SCRIPT}"
