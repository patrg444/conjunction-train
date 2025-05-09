#!/bin/bash
# Monitor the Fusion model training using the combined dataset

KEY="/Users/patrickgloria/Downloads/gpu-key.pem"
EC2_HOST="ubuntu@52.90.218.245"
LOG_FILE="/home/ubuntu/monitor_logs/fusion_training_stream.log"

echo "Streaming Fusion logs from: $EC2_HOST:$LOG_FILE"
echo "Press Ctrl+C to stop."

ssh -i $KEY $EC2_HOST "tail -f $LOG_FILE"
