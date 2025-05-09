#!/bin/bash
SSH_KEY="$HOME/Downloads/gpu-key.pem"
EC2_HOST="ubuntu@54.162.134.77"

while true; do
  echo "---------------------------------------------"
  echo "Checking training status at $(date)"
  ssh -i "$SSH_KEY" $EC2_HOST "cd /home/ubuntu/emotion_project && tail -n 20 logs/training_log.txt 2>/dev/null || echo 'No log file yet'"
  echo "---------------------------------------------"
  echo "Checking GPU status:"
  ssh -i "$SSH_KEY" $EC2_HOST "nvidia-smi"
  echo "---------------------------------------------"
  sleep 60
done
