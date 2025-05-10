#!/usr/bin/env bash
# Combined script to fix G5 training data issues and restart training
# This script:
# 1. Generates a proper laughter manifest
# 2. Uploads all necessary files to EC2
# 3. Fixes/rebuilds normalization stats if needed
# 4. Restarts training with real data

set -e  # Exit on error

# Constants
SSH_KEY="$HOME/Downloads/gpu-key.pem"
SSH_USER="ubuntu"
AWS_IP="18.208.166.91"
SSH_HOST="$SSH_USER@$AWS_IP"
S3_BUCKET="emotion-recognition-data-324037291814"
EC2_PROJECT_PATH="/home/$SSH_USER/emotion-recognition"
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
LOG_FILE="g5_fix_${TIMESTAMP}.log"

# Start logging
exec > >(tee -a "$LOG_FILE") 2>&1
echo "===== G5 Training Fix Script - $(date) ====="

# Step 1: Test SSH connection
echo "[1/8] Testing SSH connection..."
ssh -i "$SSH_KEY" -o BatchMode=yes -o ConnectTimeout=10 "$SSH_HOST" "echo Connected successfully" || {
    echo "❌ SSH connection failed."
    exit 1
}
echo "✅ SSH connection successful"

# Step 2: Generate laughter manifest
if [ -f datasets/manifests/laughter_v1.csv ]; then
  echo "[2/8] Manifest exists, skipping generation"
else
  echo "[2/8] Generating laughter manifest..."
  mkdir -p datasets/manifests
  python generate_laughter_manifest.py --output datasets/manifests/laughter_v1.csv --samples 500
  echo "✅ Generated laughter manifest"
fi

# Step 3: Stop any existing training on EC2
echo "[3/8] Stopping any existing training processes on EC2..."
ssh -q -n -i "$SSH_KEY" -o BatchMode=yes -o ConnectTimeout=10 "$SSH_HOST" \
    "pkill -f train_audio_pooling_lstm_with_laughter.py" 2>/dev/null || true
echo "✅ Completed step 3"

# Step 4: Create directories on EC2
echo "[4/8] Creating directories on EC2..."
ssh -i "$SSH_KEY" "$SSH_HOST" "mkdir -p $EC2_PROJECT_PATH/ravdess_features_facenet \
    $EC2_PROJECT_PATH/crema_d_features_facenet \
    $EC2_PROJECT_PATH/datasets/manifests \
    $EC2_PROJECT_PATH/models/dynamic_padding_no_leakage"
echo "✅ Created directories"

# Step 5: Upload scripts and manifest
if ssh -i "$SSH_KEY" "$SSH_HOST" test -f $EC2_PROJECT_PATH/fix_normalization_stats.py && ssh -i "$SSH_KEY" "$SSH_HOST" test -f $EC2_PROJECT_PATH/datasets/manifests/laughter_v1.csv; then
  echo "[5/8] Scripts and manifest exist on EC2, skipping upload"
else
  echo "[5/8] Uploading scripts and manifest..."
  scp -i "$SSH_KEY" fix_normalization_stats.py "$SSH_HOST:$EC2_PROJECT_PATH/"
  scp -i "$SSH_KEY" datasets/manifests/laughter_v1.csv "$SSH_HOST:$EC2_PROJECT_PATH/datasets/manifests/"
  echo "✅ Uploaded scripts and manifest"
fi

# Step 6: Package and upload feature archives
if [ -f ravdess_features_facenet.tar.gz ] && [ -f crema_d_features_facenet.tar.gz ]; then
  echo "[6/8] Archives exist, skipping packaging and upload"
else
  echo "[6/8] Packaging feature archives..."
  tar -czf ravdess_features_facenet.tar.gz ravdess_features_facenet
  tar -czf crema_d_features_facenet.tar.gz crema_d_features_facenet
  echo "✅ Packaged feature archives"
  echo "[6/8] Uploading archives..."
  scp -i "$SSH_KEY" ravdess_features_facenet.tar.gz "$SSH_HOST:$EC2_PROJECT_PATH/"
  scp -i "$SSH_KEY" crema_d_features_facenet.tar.gz "$SSH_HOST:$EC2_PROJECT_PATH/"
  echo "✅ Uploaded feature archives"
fi

if ssh -i "$SSH_KEY" "$SSH_HOST" test -d $EC2_PROJECT_PATH/ravdess_features_facenet && ssh -i "$SSH_KEY" "$SSH_HOST" test -d $EC2_PROJECT_PATH/crema_d_features_facenet; then
  echo "[6/8] Remote directories exist, skipping extraction"
else
  echo "[6/8] Extracting archives on EC2..."
  ssh -i "$SSH_KEY" "$SSH_HOST" bash -c "'
cd $EC2_PROJECT_PATH
echo \"[6/8] Remote: extracting RAVDESS features...\"
tar xzf ravdess_features_facenet.tar.gz
echo \"[6/8] Remote: extracting CREMA-D features...\"
tar xzf crema_d_features_facenet.tar.gz
rm ravdess_features_facenet.tar.gz crema_d_features_facenet.tar.gz
echo \"[6/8] Remote: extraction complete\"
'"
  echo "✅ Uploaded and extracted feature archives"
fi

# Step 7: Run normalization stats check/fix
if ssh -i "$SSH_KEY" "$SSH_HOST" test -f $EC2_PROJECT_PATH/models/dynamic_padding_no_leakage/audio_normalization_stats.pkl && \
   ssh -i "$SSH_KEY" "$SSH_HOST" test -f $EC2_PROJECT_PATH/models/dynamic_padding_no_leakage/video_normalization_stats.pkl; then
  echo "[7/8] Normalization stats exist on EC2, skipping"
else
  echo "[7/8] Checking and fixing normalization statistics..."
  ssh -i "$SSH_KEY" "$SSH_HOST" "cd $EC2_PROJECT_PATH && python fix_normalization_stats.py"
  echo "✅ Completed step 7"
fi

echo "[8/8] Checking Python GPU support on EC2..."
ssh -i "$SSH_KEY" "$SSH_HOST" bash -lc "python3 - << 'EOF'
import sys
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    print('TF GPUs:', gpus if gpus else 'None detected')
except Exception as e:
    print('GPU check error:', e)
EOF"
# Step 8: Restart training
echo "[8/8] Killing existing training processes on EC2..."
ssh -i "$SSH_KEY" "$SSH_HOST" "pkill -f run_audio_pooling_with_laughter.sh || true"
echo "[8/8] Restarting training..."
ssh -i "$SSH_KEY" "$SSH_HOST" bash -ilc "cd $EC2_PROJECT_PATH && mkdir -p logs && nohup bash run_audio_pooling_with_laughter.sh ${EPOCHS:-100} ${BATCH_SIZE:-256} ${MAX_SEQ_LEN:-45} ${LAUGH_WEIGHT:-0.3} > logs/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &"
echo "✅ Completed step 8"

# GPU status monitoring
echo "[GPU status] Capturing status and running process details over 3 intervals:"
ssh -i "$SSH_KEY" "$SSH_HOST" bash << 'EOF'
for i in {1..3}; do
  echo "Interval $i - $(date)"
  echo "  Util, MemUsed, MemTotal, Temp:"
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
  echo "  GPU compute processes (pid, proc, used_mem):"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  sleep 5
done
EOF

echo "============================================================"
echo "✅ G5 training fix and restart completed!"
echo "Monitor with: ./enhanced_monitor_g5.sh"
echo "============================================================"
