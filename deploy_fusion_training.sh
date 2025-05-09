#!/bin/bash
# Deploy and launch SlowFast+HuBERT fusion model training on EC2

# --- Configuration ---
KEY=~/Downloads/gpu-key.pem
# IMPORTANT: Use the correct EC2 instance IP/Host where SlowFast model and datasets reside
EC2_HOST="ubuntu@52.90.218.245" # CORRECTED IP from instance summary
SESSION_NAME="fusion_training"

# Paths on the EC2 instance
REMOTE_SCRIPT_DIR="/home/ubuntu/scripts"
REMOTE_CONFIG_DIR="/home/ubuntu/config" # Assuming config might be needed
REMOTE_OUTPUT_DIR="/home/ubuntu/emotion_fusion_output" # New output dir for fusion model
REMOTE_MANIFEST_PATH="/home/ubuntu/datasets/video_manifest.csv" # Manifest used by SlowFast/R3D
REMOTE_VIDEO_CHECKPOINT="/home/ubuntu/emotion_slowfast/slowfast_emotion_20250422_040528_best.pt" # Located R3D-18 model checkpoint (originally named slowfast)
REMOTE_HUBERT_EMBEDDINGS_DIR="/home/ubuntu/conjunction-train" # Where we copied embeddings

# Local paths
LOCAL_SCRIPT_PATH="scripts/train_fusion_model.py"

# Echo with timestamp
function log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# --- Preparations ---
log "Making local fusion script executable..."
chmod +x $LOCAL_SCRIPT_PATH

log "Checking local files..."
if [ ! -f "$LOCAL_SCRIPT_PATH" ]; then
    log "ERROR: Local fusion script not found at $LOCAL_SCRIPT_PATH"
    exit 1
fi
if [ ! -f "$KEY" ]; then
    log "ERROR: SSH key not found at $KEY"
    exit 1
fi

# --- EC2 Setup ---
log "Creating remote directories on $EC2_HOST..."
ssh -i $KEY $EC2_HOST "mkdir -p $REMOTE_SCRIPT_DIR $REMOTE_CONFIG_DIR $REMOTE_OUTPUT_DIR $REMOTE_HUBERT_EMBEDDINGS_DIR /home/ubuntu/monitor_logs"

log "Copying fusion training script to EC2..."
scp -i $KEY $LOCAL_SCRIPT_PATH $EC2_HOST:$REMOTE_SCRIPT_DIR/

log "Copying local splits directory to EC2..."
scp -r -i $KEY splits $EC2_HOST:$REMOTE_HUBERT_EMBEDDINGS_DIR/

# Verify necessary files exist on remote
log "Verifying required files on EC2..."
ssh -i $KEY $EC2_HOST "ls -lh $REMOTE_MANIFEST_PATH" || { log "ERROR: Manifest file missing on EC2: $REMOTE_MANIFEST_PATH"; exit 1; }
ssh -i $KEY $EC2_HOST "ls -lh $REMOTE_VIDEO_CHECKPOINT" || { log "ERROR: Video checkpoint missing on EC2: $REMOTE_VIDEO_CHECKPOINT"; exit 1; }
ssh -i $KEY $EC2_HOST "ls -lh ${REMOTE_HUBERT_EMBEDDINGS_DIR}/*embeddings.npz" || { log "ERROR: Hubert embeddings missing on EC2 in: $REMOTE_HUBERT_EMBEDDINGS_DIR"; exit 1; }
ssh -i $KEY $EC2_HOST "ls -lh ${REMOTE_HUBERT_EMBEDDINGS_DIR}/splits/*.csv" || { log "ERROR: Hubert split CSVs missing on EC2 in: ${REMOTE_HUBERT_EMBEDDINGS_DIR}/splits"; exit 1; }
# Check for torchvision (needed for R3D-18)
ssh -i $KEY $EC2_HOST "python3 -c 'import torchvision' || echo 'Warning: torchvision might not be installed. Install with: pip install torchvision'"


# --- Launch Training ---
log "Cleaning up any existing tmux session '$SESSION_NAME'..."
ssh -i $KEY $EC2_HOST "tmux kill-session -t $SESSION_NAME 2>/dev/null || true"

log "Launching Fusion training in tmux session '$SESSION_NAME'..."
ssh -i $KEY $EC2_HOST "tmux new-session -d -s $SESSION_NAME"

# Command to run inside tmux
# Adjust hyperparameters as needed (e.g., batch_size, lr, epochs, hubert_dim)
# Ensure torchvision is available
# Set PYTHONPATH to allow absolute import from scripts/ and run from /home/ubuntu
TRAIN_COMMAND="cd /home/ubuntu && pip install torchvision --quiet && PYTHONPATH=/home/ubuntu python3 $REMOTE_SCRIPT_DIR/train_fusion_model.py \
    --manifest_file $REMOTE_MANIFEST_PATH \
    --hubert_embeddings_dir $REMOTE_HUBERT_EMBEDDINGS_DIR \
    --hubert_splits_dir ${REMOTE_HUBERT_EMBEDDINGS_DIR}/splits \
    --video_checkpoint $REMOTE_VIDEO_CHECKPOINT \
    --output_dir $REMOTE_OUTPUT_DIR \
    --fusion_dim 512 \
    --dropout 0.5 \
    --batch_size 4 \
    --epochs 50 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --label_smoothing 0.1 \
    --early_stop 10 \
    --img_size 224 \
    --frames 48 \
    --num_workers 4 \
    --fp16" # Add --fp16 if GPU supports it and desired

ssh -i $KEY $EC2_HOST "tmux send-keys -t $SESSION_NAME '$TRAIN_COMMAND' C-m"

# --- Monitoring Setup ---
log "Setting up training logs for monitoring..."
LOG_STREAM_FILE="/home/ubuntu/monitor_logs/fusion_training_stream.log"
ssh -i $KEY $EC2_HOST "tmux pipe-pane -t $SESSION_NAME 'cat > $LOG_STREAM_FILE'"

# Create local monitor script
MONITOR_SCRIPT_NAME="monitor_${SESSION_NAME}.sh"
cat > $MONITOR_SCRIPT_NAME <<EOF
#!/bin/bash
echo "Streaming logs from EC2 session '$SESSION_NAME'..."
ssh -i $KEY $EC2_HOST "tail -f $LOG_STREAM_FILE"
EOF
chmod +x $MONITOR_SCRIPT_NAME

log "Fusion training started!"
log "Monitor full session: ssh -i $KEY $EC2_HOST \"tmux attach -t $SESSION_NAME\""
log "Monitor log stream locally: ./${MONITOR_SCRIPT_NAME}"
