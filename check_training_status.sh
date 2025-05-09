#!/bin/bash
# Script to check the status of active training jobs on the EC2 instance
# This helps determine if we should stop existing training or let it complete

# Configuration
KEY=~/Downloads/gpu-key.pem
EC2_HOST="ubuntu@54.162.134.77"

# Echo with timestamp
function log() {
    echo -e "\033[34m[$(date '+%Y-%m-%d %H:%M:%S')]\033[0m \033[1m$1\033[0m"
}

# Print header
echo -e "\033[1m\033[32m===========================================================\033[0m"
echo -e "\033[1m\033[32m          TRAINING STATUS CHECK                            \033[0m"
echo -e "\033[1m\033[32m===========================================================\033[0m"
echo ""

# Check active tmux sessions
log "Checking active tmux sessions..."
ssh -i $KEY $EC2_HOST "tmux list-sessions 2>/dev/null || echo 'No active tmux sessions'"

# Check GPU utilization
log "Checking GPU utilization..."
ssh -i $KEY $EC2_HOST "nvidia-smi || echo 'Unable to get GPU status'"

# Check for running Python processes
log "Checking for running training processes..."
ssh -i $KEY $EC2_HOST "ps aux | grep -E 'python|train' | grep -v grep || echo 'No Python training processes found'"

# Check last few lines of any training logs if they exist
log "Checking recent training logs (if any)..."
ssh -i $KEY $EC2_HOST "find /home/ubuntu -type f -name '*.log' -mtime -1 | xargs -I{} sh -c 'echo -e \"\n\033[33mFile: {}\033[0m\"; tail -n 20 {}' 2>/dev/null || echo 'No recent log files found'"

# Check disk space
log "Checking disk space..."
ssh -i $KEY $EC2_HOST "df -h | grep -E 'Filesystem|/$'"

echo ""
log "Status check complete. Based on this information, you can decide whether to:"
echo "  1. Let the current training complete if it's near the end"
echo "  2. Stop the current training and start SlowFast training"
echo ""
echo "To stop existing training sessions and start SlowFast, run:"
echo -e "\033[33m  ./deploy_complete_slowfast_pipeline.sh\033[0m"
echo ""
echo "To monitor an existing training session:"
echo -e "\033[33m  ssh -i $KEY $EC2_HOST \"tmux attach -t SESSION_NAME\"\033[0m"
echo "  (replace SESSION_NAME with the actual session name from the list above)"
