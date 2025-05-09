#!/bin/bash
# Script to deploy improved XLM-RoBERTa v3 training to EC2

# EC2 instance details
EC2_HOST="ubuntu@3.80.203.65"
KEY_PATH="/Users/patrickgloria/Downloads/gpu-key.pem"

# Check if key file exists
if [ ! -f "$KEY_PATH" ]; then
    echo "Error: SSH key file not found at $KEY_PATH"
    exit 1
fi

echo "=== Deploying improved XLM-RoBERTa v3 model to EC2 ==="
echo "EC2 Instance: $EC2_HOST"

# 1. Stop any existing XLM-RoBERTa v2 training
echo "Stopping any existing XLM-RoBERTa training processes..."
ssh -i "$KEY_PATH" $EC2_HOST "pkill -f 'python.*xlm-roberta.*' || echo 'No running XLM-RoBERTa processes found'"

# 2. Copy improved model script to EC2
echo "Copying improved XLM-RoBERTa v3 script to EC2..."
scp -i "$KEY_PATH" ./improved_xlm_roberta_v3.py $EC2_HOST:~/conjunction-train/

# 3. Start training with optimized parameters
echo "Starting improved XLM-RoBERTa v3 training..."
ssh -i "$KEY_PATH" $EC2_HOST "cd ~/conjunction-train && nohup python improved_xlm_roberta_v3.py \
    --train_manifest datasets/manifests/humor/ur_funny_train_humor_cleaned.csv \
    --val_manifest datasets/manifests/humor/ur_funny_val_humor_cleaned.csv \
    --model_name xlm-roberta-large \
    --batch_size 8 \
    --grad_accum 4 \
    --learning_rate 3e-5 \
    --epochs 50 \
    --early_stopping 5 \
    --num_workers 4 \
    --weight_decay 0.01 \
    --dropout 0.1 \
    --scheduler linear_warmup_cosine \
    --warmup_ratio 0.1 \
    --grad_clip 1.0 \
    --log_dir training_logs_humor \
    --exp_name xlm-roberta-large_v3 \
    --class_balancing \
    --label_smoothing 0.1 \
    --layerwise_lr_decay 0.95 \
    --fp16 \
    --monitor_metric val_f1 \
    --seed 42 \
    --devices 1 \
    > xlm_roberta_v3_training.log 2>&1 &"

echo "Verifying training started..."
sleep 5
ssh -i "$KEY_PATH" $EC2_HOST "ps aux | grep improved_xlm_roberta_v3.py | grep -v grep"

if [ $? -eq 0 ]; then
    echo "Training successfully started!"
    echo "Log file: xlm_roberta_v3_training.log"
    echo ""
    echo "To monitor training, run:"
    echo "./monitor_xlm_roberta_v3.sh"
else
    echo "Error: Training did not start successfully. Check EC2 instance."
fi

# 4. Create monitoring script
cat > monitor_xlm_roberta_v3.sh << 'EOF'
#!/bin/bash
# Script to monitor XLM-RoBERTa v3 training

# EC2 instance details
EC2_HOST="ubuntu@3.80.203.65"
KEY_PATH="/Users/patrickgloria/Downloads/gpu-key.pem"

# Check if key file exists
if [ ! -f "$KEY_PATH" ]; then
    echo "Error: SSH key file not found at $KEY_PATH"
    exit 1
fi

echo "== XLM-RoBERTa v3 Training Monitor ==="
echo "EC2 Instance IP: 3.80.203.65"
echo "Monitoring log file: /home/ubuntu/conjunction-train/xlm_roberta_v3_training.log"
echo ""

echo "1. Checking if training is running..."
ssh -i "$KEY_PATH" $EC2_HOST "ps aux | grep improved_xlm_roberta_v3.py | grep -v grep"

if [ $? -eq 0 ]; then
    echo "Training is currently running!"
else
    echo "Training is not running!"
fi

echo ""
echo "2. Displaying the last 50 lines of the log file:"
echo "-------------------------------------------------------------------------------"
ssh -i "$KEY_PATH" $EC2_HOST "tail -n 50 ~/conjunction-train/xlm_roberta_v3_training.log"
echo "-------------------------------------------------------------------------------"

echo ""
echo "3. Options for monitoring:"
echo "a) To follow the log in real-time, run:"
echo "   ssh -i \"$KEY_PATH\" $EC2_HOST \"tail -f /home/ubuntu/conjunction-train/xlm_roberta_v3_training.log\""
echo ""
echo "b) To check GPU usage, run:"
echo "   ssh -i \"$KEY_PATH\" $EC2_HOST \"nvidia-smi\""
echo ""
echo "c) To check the best model saved so far, run:"
echo "   ssh -i \"$KEY_PATH\" $EC2_HOST \"ls -la /home/ubuntu/conjunction-train/training_logs_humor/xlm-roberta-large_v3/checkpoints/\""
echo ""
echo "Monitor complete!"
EOF

chmod +x monitor_xlm_roberta_v3.sh
echo "Created monitoring script: monitor_xlm_roberta_v3.sh"

echo "Deployment complete!"
