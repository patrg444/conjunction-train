#!/bin/bash
# Script to monitor emotion comparison jobs running on EC2
set -e

# Configure paths and credentials
KEY=~/Downloads/gpu-key.pem
EC2=ubuntu@54.162.134.77
REMOTE_DIR=/home/ubuntu/emotion_cmp

echo "======================================================="
echo "🔍 Monitoring Emotion Comparison Jobs on EC2"
echo "======================================================="
echo "Target: $EC2:$REMOTE_DIR"
echo "SSH Key: $KEY"
echo

# Check system resources
echo "📊 System Resources:"
ssh -i $KEY $EC2 "echo 'CPU Usage:'; top -bn1 | head -15; echo; echo 'Memory Usage:'; free -h; echo; echo 'GPU Status:'; nvidia-smi | head -20"
echo

# Check for running jobs
echo "🔄 Running Processes:"
ssh -i $KEY $EC2 "ps aux | grep -E 'run_emotion_comparison|python' | grep -v grep || echo 'No emotion comparison jobs currently running'"
echo

# Check for completed results
echo "📋 Comparison Results Status:"
ssh -i $KEY $EC2 "echo 'RAVDESS Results:'; find $REMOTE_DIR/comparison_results -name 'RAVDESS*' -type d -exec ls -la {} \; 2>/dev/null || echo 'No RAVDESS results found'; 
                  echo; echo 'CREMA-D Results:'; find $REMOTE_DIR/comparison_results -name 'CREMAD*' -type d -exec ls -la {} \; 2>/dev/null || echo 'No CREMA-D results found'"
echo

# Show most recent log files (if any)
echo "📜 Recent Logs:"
ssh -i $KEY $EC2 "find $REMOTE_DIR/comparison_results -name '*.log' -type f -exec ls -lt {} \; | head -5 || echo 'No log files found'"
echo

# If there are log files, show the tail of the most recent one
echo "📝 Latest Log Output:"
LATEST_LOG=$(ssh -i $KEY $EC2 "find $REMOTE_DIR/comparison_results -name '*.log' -type f -exec ls -t {} \; | head -1 || echo ''")
if [ -n "$LATEST_LOG" ]; then
  ssh -i $KEY $EC2 "tail -n 30 $LATEST_LOG"
else
  echo "No log files found."
fi
echo

# Provide commands to start comparison jobs
echo "======================================================="
echo "✅ Monitoring Complete"
echo "======================================================="
echo
echo "To start a comparison job on EC2:"
echo "  ssh -i $KEY $EC2"
echo "  cd $REMOTE_DIR"
echo "  nohup ./run_emotion_comparison.sh --ravdess --auto > ravdess_comparison.log 2>&1 &    # For RAVDESS"
echo "  nohup ./run_emotion_comparison.sh --cremad --auto > cremad_comparison.log 2>&1 &      # For CREMA-D"
echo
echo "To run this monitoring script again:"
echo "  ./monitor_emotion_comparison.sh"
echo
echo "To download results after completion:"
echo "  rsync -avP -e \"ssh -i $KEY\" $EC2:$REMOTE_DIR/comparison_results/ ./comparison_results_ec2/"
echo
