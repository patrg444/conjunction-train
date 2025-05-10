#!/usr/bin/env bash
# Enhanced monitoring script for G5 GPU training with TensorBoard setup

SSH_KEY="$HOME/Downloads/gpu-key.pem"
SSH_USER="ubuntu"
AWS_IP="18.208.166.91"
SSH_HOST="$SSH_USER@$AWS_IP"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TRAIN_LOG="train_g5_fixed_20250419_132500.log"
TENSORBOARD_PORT=6006

echo "======================================"
echo "     G5 TRAINING MONITOR SCRIPT      "
echo "======================================"

# Check if the training process is still running
echo -e "\n[1/6] Checking if training process is running..."
IS_RUNNING=$(ssh -i "$SSH_KEY" "$SSH_HOST" "pgrep -f train_audio_pooling_lstm_fixed.py || echo 'not_running'")
if [[ "$IS_RUNNING" == "not_running" ]]; then
    echo "⚠️  WARNING: Training process is not currently running!"
else
    echo "✅ Training process is active (PID: $IS_RUNNING)"
fi

# Check data presence and sizes
echo -e "\n[2/6] Checking dataset presence and sizes..."
ssh -i "$SSH_KEY" "$SSH_HOST" "cd ~/emotion-recognition && \
    echo '- RAVDESS features:' && du -sh ravdess_features_facenet/ 2>/dev/null && \
    echo '- CREMA-D features:' && du -sh crema_d_features_facenet/ 2>/dev/null && \
    echo '- Normalization files:' && find models -name '*_normalization_stats.pkl' | xargs ls -lh 2>/dev/null || echo 'No normalization files found'"

# Check GPU status with improved output
echo -e "\n[3/6] Checking GPU status..."
ssh -i "$SSH_KEY" "$SSH_HOST" "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader"

# Check for TensorBoard and set it up if not already running
echo -e "\n[4/6] Checking/setting up TensorBoard..."
TB_RUNNING=$(ssh -i "$SSH_KEY" "$SSH_HOST" "pgrep -f tensorboard || echo 'not_running'")
if [[ "$TB_RUNNING" == "not_running" ]]; then
    echo "TensorBoard not running. Setting up..."
    
    # Create TensorBoard logs directory if it doesn't exist
    ssh -i "$SSH_KEY" "$SSH_HOST" "mkdir -p ~/emotion-recognition/logs/tensorboard"
    
    # Launch TensorBoard in the background
    ssh -i "$SSH_KEY" "$SSH_HOST" "cd ~/emotion-recognition && nohup tensorboard --logdir=logs/tensorboard --port=$TENSORBOARD_PORT --host=localhost > ~/tensorboard.log 2>&1 &"
    
    echo "TensorBoard started. To view it, open a new terminal and run:"
    echo "ssh -i $SSH_KEY -L $TENSORBOARD_PORT:localhost:$TENSORBOARD_PORT $SSH_HOST"
    echo "Then open http://localhost:$TENSORBOARD_PORT in your browser"
else
    echo "✅ TensorBoard is already running (PID: $TB_RUNNING)"
    echo "To view it, open a new terminal and run:"
    echo "ssh -i $SSH_KEY -L $TENSORBOARD_PORT:localhost:$TENSORBOARD_PORT $SSH_HOST"
    echo "Then open http://localhost:$TENSORBOARD_PORT in your browser"
fi

# Show training progress
echo -e "\n[5/6] Showing recent training log (last 50 lines)..."
ssh -i "$SSH_KEY" "$SSH_HOST" "tail -n 50 ~/$TRAIN_LOG"

# Calculate estimated training time
echo -e "\n[6/6] Calculating estimated training progress..."
START_TIME=$(ssh -i "$SSH_KEY" "$SSH_HOST" "stat -c %Y ~/$TRAIN_LOG")
CURRENT_TIME=$(date +%s)
ELAPSED_SECONDS=$((CURRENT_TIME - START_TIME))
ELAPSED_HOURS=$(echo "scale=1; $ELAPSED_SECONDS/3600" | bc)

# Try to extract current epoch
CURRENT_EPOCH=$(ssh -i "$SSH_KEY" "$SSH_HOST" "grep -o 'Epoch [0-9]*/100' ~/$TRAIN_LOG | tail -1 | cut -d' ' -f2 | cut -d'/' -f1")
if [[ -n "$CURRENT_EPOCH" ]]; then
    PROGRESS_PCT=$(echo "scale=1; $CURRENT_EPOCH/100*100" | bc)
    AVG_EPOCH_TIME=$(echo "scale=2; $ELAPSED_SECONDS/$CURRENT_EPOCH" | bc 2>/dev/null)
    REMAINING_EPOCHS=$((100 - CURRENT_EPOCH))
    REMAINING_SECONDS=$(echo "$AVG_EPOCH_TIME * $REMAINING_EPOCHS" | bc 2>/dev/null)
    REMAINING_HOURS=$(echo "scale=1; $REMAINING_SECONDS/3600" | bc)
    
    echo "Training progress: Epoch $CURRENT_EPOCH/100 ($PROGRESS_PCT%)"
    echo "Running for: $ELAPSED_HOURS hours"
    echo "Estimated time remaining: $REMAINING_HOURS hours"
    echo "Estimated completion: $(date -d "+$REMAINING_SECONDS seconds" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || date -v "+${REMAINING_SECONDS}S" "+%Y-%m-%d %H:%M:%S")"
else
    echo "Training progress: Unable to determine current epoch"
    echo "Running for: $ELAPSED_HOURS hours"
fi

echo -e "\nRun this script periodically to monitor training progress."
echo "When training is complete, use ./download_g5_fixed_model_20250419_132500.sh to download the model."
