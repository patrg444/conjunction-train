#!/bin/bash
# Deploy and launch SlowFast emotion recognition training on EC2
# This script copies the necessary files to the EC2 instance and starts training in a tmux session

# Configuration
KEY=~/Downloads/gpu-key.pem
EC2_HOST="ubuntu@54.162.134.77"
SESSION_NAME="slowfast_training"
MANIFEST_PATH="/home/ubuntu/datasets/video_manifest.csv"
OUTPUT_DIR="/home/ubuntu/emotion_slowfast"

# Echo with timestamp
function log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Make sure training script is executable
chmod +x scripts/train_slowfast_emotion.py

# Create remote directories
log "Creating remote directories..."
ssh -i $KEY $EC2_HOST "mkdir -p /home/ubuntu/scripts /home/ubuntu/config /home/ubuntu/emotion_slowfast /home/ubuntu/monitor_logs"

# Copy necessary files
log "Copying SlowFast training files to EC2..."
scp -i $KEY scripts/train_slowfast_emotion.py $EC2_HOST:/home/ubuntu/scripts/
scp -i $KEY config/slowfast_face.yaml $EC2_HOST:/home/ubuntu/config/

# Check if manifest file exists on remote server
log "Checking manifest file..."
ssh -i $KEY $EC2_HOST "ls -la $MANIFEST_PATH || echo 'Manifest file not found!'"

# Kill existing session if running
log "Cleaning up any existing sessions..."
ssh -i $KEY $EC2_HOST "tmux kill-session -t $SESSION_NAME 2>/dev/null || true"

# Create new tmux session and start training
log "Launching SlowFast training in tmux session..."
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

# Create monitoring stream
log "Setting up training logs for monitoring..."
ssh -i $KEY $EC2_HOST "mkdir -p /home/ubuntu/monitor_logs"
ssh -i $KEY $EC2_HOST "tmux pipe-pane -t $SESSION_NAME 'cat > /home/ubuntu/monitor_logs/slowfast_training_stream.log'"

# Launch continuous monitoring in separate tmux session
ssh -i $KEY $EC2_HOST "tmux new-session -d -s slowfast_monitor \
    'tail -f /home/ubuntu/monitor_logs/slowfast_training_stream.log | grep -E \"(Epoch|Train Loss|Val Loss|accuracy)\"'"

log "SlowFast training started! Monitor with:"
echo "   ssh -i $KEY $EC2_HOST \"tmux attach -t $SESSION_NAME\"  # Full session"
echo "   ssh -i $KEY $EC2_HOST \"tmux attach -t slowfast_monitor\"  # Just progress"
echo "   ssh -i $KEY $EC2_HOST \"tail -f /home/ubuntu/monitor_logs/slowfast_training_stream.log\"  # From local"

# Create local monitor script
cat > monitor_slowfast_training.sh <<EOF
#!/bin/bash
ssh -i $KEY $EC2_HOST "tail -f /home/ubuntu/monitor_logs/slowfast_training_stream.log"
EOF
chmod +x monitor_slowfast_training.sh

log "Local monitoring script created: ./monitor_slowfast_training.sh"
