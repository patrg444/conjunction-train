#!/bin/bash
# Helper script for monitoring Facenet training on AWS GPU

SSH_KEY="${SSH_KEY:-/Users/patrickgloria/Downloads/gpu-key.pem}"
INSTANCE_IP="${1:-18.208.166.91}"

if [[ -z "$INSTANCE_IP" ]]; then
  echo "Usage: $0 <instance-ip>"
  echo "Or set INSTANCE_IP environment variable"
  exit 1
fi

echo "=== Facenet Training Monitor Helper ==="
echo "SSH key: $SSH_KEY"
echo "Instance IP: $INSTANCE_IP"
echo ""

# Check if training directory exists
echo "Checking if training directory exists..."
DIR_CHECK=$(ssh -i "$SSH_KEY" ubuntu@"$INSTANCE_IP" "ls -la /home/ubuntu/emotion-recognition/facenet_full_training 2>/dev/null || echo 'DIRECTORY_NOT_FOUND'")

if [[ "$DIR_CHECK" == *"DIRECTORY_NOT_FOUND"* ]]; then
  echo "❌ Error: The training directory does not exist!"
  echo "    Expected: /home/ubuntu/emotion-recognition/facenet_full_training"
  echo ""
  echo "Let's check where the training files might be located:"
  ssh -i "$SSH_KEY" ubuntu@"$INSTANCE_IP" "find /home/ubuntu -name 'train_facenet_full.py' -o -name 'fixed_video_facenet_generator.py' 2>/dev/null"
  exit 1
fi

# Check if training is running
echo "Checking if training is running..."
PROCESS_CHECK=$(ssh -i "$SSH_KEY" ubuntu@"$INSTANCE_IP" "ps aux | grep train_facenet_full.py | grep -v grep || echo 'PROCESS_NOT_FOUND'")

if [[ "$PROCESS_CHECK" == *"PROCESS_NOT_FOUND"* ]]; then
  echo "❌ Warning: No running training process found. The training might not have started or might have terminated."
else
  echo "✅ Training process is running!"
  echo "$PROCESS_CHECK"
fi

# Check tmux sessions
echo "Checking tmux sessions..."
TMUX_CHECK=$(ssh -i "$SSH_KEY" ubuntu@"$INSTANCE_IP" "tmux list-sessions 2>/dev/null || echo 'NO_TMUX_SESSIONS'")

if [[ "$TMUX_CHECK" == *"NO_TMUX_SESSIONS"* ]]; then
  echo "❌ Warning: No tmux sessions found. The training might not be running in tmux."
else
  echo "✅ Tmux sessions found:"
  echo "$TMUX_CHECK"
fi

echo ""
echo "=== Correct Commands for Monitoring ==="
echo ""
echo "To SSH into the server:"
echo "  ssh -i $SSH_KEY ubuntu@$INSTANCE_IP"
echo ""
echo "Once logged in, to check the training directory:"
echo "  ls -la /home/ubuntu/emotion-recognition/facenet_full_training"
echo ""
echo "To find all Python training scripts:"
echo "  find /home/ubuntu -name \"*.py\" | grep -E \"facenet|train\""
echo ""
echo "To check GPU usage:"
echo "  nvidia-smi"
echo ""
echo "To attach to the tmux session (if it exists):"
echo "  tmux attach -t facenet_training"
echo ""
echo "To set up TensorBoard (from your local machine):"
echo "  ssh -i $SSH_KEY -L 6006:localhost:6006 ubuntu@$INSTANCE_IP"
echo "  # Then on the server:"
echo "  source ~/facenet-venv/bin/activate"
echo "  cd /path/to/training/directory"
echo "  tensorboard --logdir=logs"
echo ""
echo "Monitor Helper script finished"
