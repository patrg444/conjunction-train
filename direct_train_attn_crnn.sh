#!/bin/bash
#
# Direct script to run training with full error logging
#

set -euo pipefail

# Configuration
IP="54.162.134.77"
PEM="$HOME/Downloads/gpu-key.pem"

echo "=== DIRECTLY RUNNING ATTN-CRNN TRAINING WITH OUTPUT CAPTURE ==="

# Create a simple directory structure for the WA2VEC features if not already there
ssh -i "$PEM" ubuntu@$IP << 'EOF'
  set -euo pipefail

  # Ensure the directories exist
  mkdir -p /home/ubuntu/emotion_project/scripts
  mkdir -p /home/ubuntu/emotion_project/logs

  # Copy the training script to the scripts directory if it doesn't exist
  if [ ! -f "/home/ubuntu/emotion_project/scripts/train_attn_crnn.py" ]; then
    if [ -f "/home/ubuntu/scripts/train_attn_crnn.py" ]; then
      cp /home/ubuntu/scripts/train_attn_crnn.py /home/ubuntu/emotion_project/scripts/
    else
      find /home/ubuntu -name "train_attn_crnn.py" -exec cp {} /home/ubuntu/emotion_project/scripts/ \; 2>/dev/null
    fi
  fi

  # Ensure symlink to wav2vec features exists
  if [ ! -L "/home/ubuntu/emotion_project/wav2vec_features" ]; then
    ln -sf "/home/ubuntu/audio_emotion/models/wav2vec" /home/ubuntu/emotion_project/wav2vec_features
  fi

  # Create a very simple direct script that runs the training with error capture
  cat > /home/ubuntu/direct_train.sh << 'SHELL'
#!/bin/bash
cd /home/ubuntu
source /opt/pytorch/bin/activate

export PYTHONPATH=/home/ubuntu/emotion_project:${PYTHONPATH:-}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/home/ubuntu/emotion_project/logs/direct_train_${TIMESTAMP}.log"

echo "=== Starting direct training at $(date) ===" | tee -a $LOG_FILE

# Check feature files
echo "Checking WAV2VEC feature files..." | tee -a $LOG_FILE
FEATURE_COUNT=$(find -L /home/ubuntu/emotion_project/wav2vec_features -name "*.npz" | wc -l)
echo "Found $FEATURE_COUNT feature files" | tee -a $LOG_FILE

# Add verbose output for debugging
echo "Launching training script with verbose output:" | tee -a $LOG_FILE
echo "Python: $(which python)" | tee -a $LOG_FILE
echo "TensorFlow: $(python -c 'import tensorflow as tf; print(tf.__version__)')" | tee -a $LOG_FILE

# Run with full error capture
set -x
python -u /home/ubuntu/emotion_project/scripts/train_attn_crnn.py \
  --data_dirs=/home/ubuntu/emotion_project/wav2vec_features \
  --epochs=50 \
  --batch_size=32 \
  --lr=0.001 \
  --verbose=1 \
  --model_save_path=/home/ubuntu/emotion_project/best_attn_crnn_model.h5 2>&1 | tee -a $LOG_FILE
SHELL

  chmod +x /home/ubuntu/direct_train.sh
  
  # Run directly to get output
  echo "Starting training with direct output capture..."
  /home/ubuntu/direct_train.sh
EOF

echo "=== DIRECT TRAINING COMPLETE ==="
