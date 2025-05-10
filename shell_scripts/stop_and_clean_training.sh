#!/bin/bash
# Stop SlowFast training and clean up

KEY=~/Downloads/gpu-key.pem
EC2_HOST="ubuntu@54.162.134.77"
SESSION_NAME="slowfast_training"
MONITOR_SESSION="slowfast_monitor"

echo -e "\033[1m\033[33m===========================================================\033[0m"
echo -e "\033[1m\033[33m          STOPPING SLOWFAST TRAINING SESSION               \033[0m"
echo -e "\033[1m\033[33m===========================================================\033[0m"
echo ""

# Stop training sessions
echo "Stopping existing training sessions..."
ssh -i $KEY $EC2_HOST "tmux kill-session -t $SESSION_NAME 2>/dev/null || true"
ssh -i $KEY $EC2_HOST "tmux kill-session -t $MONITOR_SESSION 2>/dev/null || true"

# Clean up monitoring logs
echo "Cleaning up monitoring logs..."
ssh -i $KEY $EC2_HOST "rm -f /home/ubuntu/monitor_logs/slowfast_training_stream.log"

echo "Training stopped. Ready to redeploy with fixed code."
