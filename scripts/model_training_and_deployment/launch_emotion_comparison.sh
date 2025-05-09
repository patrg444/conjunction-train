#!/bin/bash
# Script to launch emotion comparison jobs on EC2
set -e

# Configure paths and credentials
KEY=~/Downloads/gpu-key.pem
EC2=ubuntu@54.162.134.77
REMOTE_DIR=/home/ubuntu/emotion_cmp

echo "======================================================="
echo "🚀 Launching Emotion Comparison Job on EC2"
echo "======================================================="
echo "Target: $EC2:$REMOTE_DIR"
echo "SSH Key: $KEY"
echo

# Check if dataset argument is provided
if [ "$#" -lt 1 ]; then
    echo "❌ Error: Missing dataset argument"
    echo "Usage: $0 [ravdess|cremad]"
    echo "  Example: $0 ravdess"
    echo "  Example: $0 cremad"
    exit 1
fi

# Parse dataset argument
DATASET=$(echo "$1" | tr '[:upper:]' '[:lower:]')
if [ "$DATASET" != "ravdess" ] && [ "$DATASET" != "cremad" ]; then
    echo "❌ Error: Invalid dataset. Must be 'ravdess' or 'cremad'"
    echo "Usage: $0 [ravdess|cremad]"
    exit 1
fi

# Format for display
if [ "$DATASET" == "ravdess" ]; then
    DISPLAY_DATASET="RAVDESS"
    DATASET_ARG="--ravdess"
else
    DISPLAY_DATASET="CREMA-D"
    DATASET_ARG="--cremad"
fi

echo "📊 Selected Dataset: $DISPLAY_DATASET"
echo

# Launch the comparison job on EC2
echo "🔄 Starting $DISPLAY_DATASET comparison job..."
ssh -i $KEY $EC2 "cd $REMOTE_DIR && nohup ./run_emotion_comparison.sh $DATASET_ARG --auto > ${DATASET}_comparison.log 2>&1 &"
echo "✅ Job started in the background"
echo

# Wait a moment for the job to initialize
echo "⏳ Waiting 3 seconds for the job to initialize..."
sleep 3
echo

# Show initial job status
echo "📋 Initial Job Status:"
ssh -i $KEY $EC2 "ps aux | grep -E 'run_emotion_comparison|python' | grep -v grep"
echo

echo "======================================================="
echo "✅ Job Launch Complete"
echo "======================================================="
echo
echo "The $DISPLAY_DATASET comparison job is now running on the EC2 instance."
echo "It will continue running in the background even after you disconnect."
echo
echo "To monitor the job's progress:"
echo "  ./monitor_emotion_comparison.sh"
echo
echo "To download results after completion:"
echo "  rsync -avP -e \"ssh -i $KEY\" $EC2:$REMOTE_DIR/comparison_results/ ./comparison_results_ec2/"
echo
