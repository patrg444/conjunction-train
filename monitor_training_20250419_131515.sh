#!/usr/bin/env bash
# Monitor script to track training on AWS instance

SSH_KEY="$HOME/Downloads/gpu-key.pem"
SSH_USER="ubuntu"
AWS_IP="18.208.166.91"
SSH_HOST="$SSH_USER@$AWS_IP"

# Set up monitoring daemon on remote server if it doesn't exist
ssh -i "$SSH_KEY" "$SSH_HOST" "mkdir -p ~/emotion-recognition/logs/monitoring"
ssh -i "$SSH_KEY" "$SSH_HOST" "if [ ! -f ~/emotion-recognition/monitor.pid ]; then
  cd ~/emotion-recognition
  nohup bash -c 'while true; do date >> logs/monitoring/monitor.log; \
    find logs -name \"train_laugh_*.log\" -exec tail -n 10 {} \; >> logs/monitoring/monitor.log 2>&1; \
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv >> logs/monitoring/monitor.log 2>&1; \
    echo \"--------------------------\" >> logs/monitoring/monitor.log; sleep 60; \
  done' > /dev/null 2>&1 &
  echo \$! > monitor.pid
  echo \"Started monitoring daemon with PID: \$(cat monitor.pid)\"
fi"

# Monitor training logs
echo "Connecting to EC2 instance to view monitoring logs..."
ssh -i "$SSH_KEY" "$SSH_HOST" "tail -f ~/emotion-recognition/logs/monitoring/monitor.log"
