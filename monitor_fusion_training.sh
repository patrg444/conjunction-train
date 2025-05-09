#!/bin/bash
echo "Streaming logs from EC2 session 'fusion_training'..."
ssh -i /Users/patrickgloria/Downloads/gpu-key.pem ubuntu@52.90.218.245 "tail -f /home/ubuntu/monitor_logs/fusion_training_stream.log"
