#!/bin/bash

echo "=== Monitoring Video-Only Facenet + LSTM Training (Fixed Version) ==="
echo "Log file: /home/ubuntu/emotion-recognition/video_only_facenet_lstm_fixed.log"
echo "Checkpoint pattern: /home/ubuntu/emotion-recognition/models/video_only_facenet_lstm_*"
echo "Connecting to 18.208.166.91..."
echo "Press Ctrl+C to stop monitoring."
echo

# Monitor for new checkpoint directories
function monitor_checkpoints() {
  ssh -i ~/Downloads/gpu-key.pem ubuntu@18.208.166.91 "ls -dt /home/ubuntu/emotion-recognition/models/video_only_facenet_lstm_* 2>/dev/null | head -1 || echo 'No checkpoint directory found yet.'"
}

# Start continuous monitoring
while true; do
  echo "--- Latest Checkpoint Directory ---"
  monitor_checkpoints
  echo
  
  # Tail the log file
  echo "--- Recent Training Log ---"
  ssh -i ~/Downloads/gpu-key.pem ubuntu@18.208.166.91 "tail -20 /home/ubuntu/emotion-recognition/video_only_facenet_lstm_fixed.log"
  echo
  
  # Sleep for a few seconds before checking again
  sleep 5
done
