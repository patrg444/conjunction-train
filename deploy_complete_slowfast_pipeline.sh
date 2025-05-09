#!/bin/bash
# Complete SlowFast Emotion Recognition Pipeline
# This script executes the full pipeline:
# 1. Generate dataset manifest
# 2. Deploy and start SlowFast training
# 3. Set up monitoring

# Configuration
KEY=~/Downloads/gpu-key.pem
EC2_HOST="ubuntu@54.162.134.77"
RAVDESS_DIR="/home/ubuntu/datasets/ravdess_videos"
CREMA_DIR="/home/ubuntu/datasets/crema_d_videos"
MANIFEST_PATH="/home/ubuntu/datasets/video_manifest.csv"
OUTPUT_DIR="/home/ubuntu/emotion_slowfast"
SESSION_NAME="slowfast_training"
MONITOR_SESSION="slowfast_monitor"

# Print header
echo -e "\033[1m\033[32m===========================================================\033[0m"
echo -e "\033[1m\033[32m          SLOWFAST EMOTION RECOGNITION PIPELINE            \033[0m"
echo -e "\033[1m\033[32m===========================================================\033[0m"
echo ""

# Echo with timestamp
function log() {
    echo -e "\033[34m[$(date '+%Y-%m-%d %H:%M:%S')]\033[0m \033[1m$1\033[0m"
}

# Check if files exist
for file in scripts/generate_video_manifest.py scripts/train_slowfast_emotion.py config/slowfast_face.yaml; do
    if [ ! -f "$file" ]; then
        echo -e "\033[31mError: Required file $file not found!\033[0m"
        exit 1
    fi
done

# Make sure scripts are executable
chmod +x scripts/generate_video_manifest.py
chmod +x scripts/train_slowfast_emotion.py

# Step 1: Check remote dataset directories
log "Step 1: Verifying datasets on EC2"
ssh -i $KEY $EC2_HOST "ls -la $RAVDESS_DIR" > /dev/null || { echo -e "\033[31mError: RAVDESS directory not found!\033[0m"; exit 1; }
ssh -i $KEY $EC2_HOST "ls -la $CREMA_DIR" > /dev/null || { echo -e "\033[31mError: CREMA-D directory not found!\033[0m"; exit 1; }

# Create remote script directory if needed
ssh -i $KEY $EC2_HOST "mkdir -p /home/ubuntu/scripts /home/ubuntu/config /home/ubuntu/emotion_slowfast /home/ubuntu/monitor_logs"

# Step 2: Generate manifest file
log "Step 2: Generating dataset manifest"
log "Copying manifest generator script to EC2..."
scp -i $KEY scripts/generate_video_manifest.py $EC2_HOST:/home/ubuntu/scripts/

log "Running manifest generation..."
ssh -i $KEY $EC2_HOST "python /home/ubuntu/scripts/generate_video_manifest.py \
    --ravdess_dir $RAVDESS_DIR \
    --crema_dir $CREMA_DIR \
    --output $MANIFEST_PATH"

# Verify manifest was created
ssh -i $KEY $EC2_HOST "ls -la $MANIFEST_PATH" > /dev/null || { echo -e "\033[31mError: Manifest file creation failed!\033[0m"; exit 1; }
MANIFEST_COUNT=$(ssh -i $KEY $EC2_HOST "wc -l $MANIFEST_PATH | awk '{print \$1}'")
log "Manifest created with $MANIFEST_COUNT entries"

# Step 3: Copy SlowFast training files to EC2
log "Step 3: Deploying SlowFast training files"
scp -i $KEY scripts/train_slowfast_emotion.py $EC2_HOST:/home/ubuntu/scripts/
scp -i $KEY config/slowfast_face.yaml $EC2_HOST:/home/ubuntu/config/

# Step 4: Kill existing session if running
log "Step 4: Preparing training environment"
ssh -i $KEY $EC2_HOST "tmux kill-session -t $SESSION_NAME 2>/dev/null || true"
ssh -i $KEY $EC2_HOST "tmux kill-session -t $MONITOR_SESSION 2>/dev/null || true"

# Step 5: Create new tmux session and start training
log "Step 5: Launching SlowFast training"
ssh -i $KEY $EC2_HOST "tmux new-session -d -s $SESSION_NAME"
ssh -i $KEY $EC2_HOST "tmux send-keys -t $SESSION_NAME 'cd /home/ubuntu && python3 scripts/train_slowfast_emotion.py \
    --manifest_file $MANIFEST_PATH \
    --output_dir $OUTPUT_DIR \
    --config /home/ubuntu/config/slowfast_face.yaml \
    --batch_size 6 \
    --epochs 60 \
    --img_size 112 \
    --frames 48 \
    --num_workers 4 \
    --clips_per_video 2 \
    --fp16' C-m"

# Step 6: Create monitoring stream
log "Step 6: Setting up training logs for monitoring"
ssh -i $KEY $EC2_HOST "mkdir -p /home/ubuntu/monitor_logs"
ssh -i $KEY $EC2_HOST "tmux pipe-pane -t $SESSION_NAME 'cat > /home/ubuntu/monitor_logs/slowfast_training_stream.log'"

# Step 7: Launch continuous monitoring in separate tmux session
log "Step 7: Starting monitoring session"
ssh -i $KEY $EC2_HOST "tmux new-session -d -s $MONITOR_SESSION \
    'tail -f /home/ubuntu/monitor_logs/slowfast_training_stream.log | grep -E \"(Epoch|Train Loss|Val Loss|accuracy)\"'"

# Step 8: Print monitoring instructions
log "Pipeline successfully deployed! Monitor with:"
echo -e "\033[33m   ./monitor_slowfast_progress.sh                  # Enhanced monitoring from local machine\033[0m"
echo -e "\033[33m   ssh -i $KEY $EC2_HOST \"tmux attach -t $SESSION_NAME\"     # Full training session view\033[0m"
echo -e "\033[33m   ssh -i $KEY $EC2_HOST \"tmux attach -t $MONITOR_SESSION\"  # Condensed metrics view\033[0m"

# Create timestamp for checking training progress later
echo $(date +%s) > .slowfast_training_started

log "Expected completion time: ~24 hours from now"
log "To download the trained model later, run: ./download_slowfast_model.sh"
