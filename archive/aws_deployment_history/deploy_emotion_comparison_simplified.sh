#!/bin/bash
# Simplified deployment script for emotion comparison framework to EC2
set -e

# Configure paths and credentials
KEY=~/Downloads/gpu-key.pem
EC2=ubuntu@54.162.134.77
SRC=$(pwd)  # Current directory - emotion recognition project root
REMOTE_DIR=/home/ubuntu/emotion_cmp

echo "======================================================="
echo "🚀 Deploying Emotion Comparison Framework to EC2"
echo "======================================================="
echo "Source: $SRC"
echo "Target: $EC2:$REMOTE_DIR"
echo "SSH Key: $KEY"
echo

# Step 1: Create remote directory
echo "📁 Creating remote directory structure..."
ssh -i $KEY $EC2 "mkdir -p $REMOTE_DIR/comparison_results"

# Step 2: Sync only necessary code
echo "📤 Syncing emotion comparison code to EC2..."
rsync -avP \
      -e "ssh -i $KEY" \
      "$SRC/emotion_comparison/" "$EC2:$REMOTE_DIR/emotion_comparison/"

echo "📤 Syncing run script..."
rsync -avP -e "ssh -i $KEY" \
      "$SRC/run_emotion_comparison.sh" \
      "$EC2:$REMOTE_DIR/"

# Step 3: Verify datasets are present
echo "🔍 Verifying datasets on EC2..."
ssh -i $KEY $EC2 <<'REMOTE_VERIFY'
if [ -d "/home/ubuntu/datasets/ravdess_videos" ]; then
    echo "✅ RAVDESS dataset found with $(find /home/ubuntu/datasets/ravdess_videos -type f | wc -l) files"
else
    echo "❌ RAVDESS dataset not found at /home/ubuntu/datasets/ravdess_videos"
    exit 1
fi

if [ -d "/home/ubuntu/datasets/crema_d_videos" ]; then
    echo "✅ CREMA-D dataset found with $(find /home/ubuntu/datasets/crema_d_videos -type f | wc -l) files"
else
    echo "❌ CREMA-D dataset not found at /home/ubuntu/datasets/crema_d_videos"
    exit 1
fi
REMOTE_VERIFY

# Step 4: Create run script with correct paths for EC2
echo "📝 Creating run script for EC2..."
ssh -i $KEY $EC2 "cat > ~/emotion_cmp/run_ec2.sh" << 'REMOTE_RUN'
#!/bin/bash
# Run script for the emotion comparison framework on EC2
set -e

# Activate Python environment (assume it's already set up)
# If you need to create a new environment, uncomment these lines:
# python3 -m venv ~/venv
# source ~/venv/bin/activate
# pip install scikit-learn matplotlib pandas seaborn

cd ~/emotion_cmp

# Create symbolic links to the datasets
mkdir -p downsampled_videos/RAVDESS
mkdir -p downsampled_videos/CREMA-D-audio-complete

echo "Creating symbolic links to datasets..."
ln -sf /home/ubuntu/datasets/ravdess_videos/* downsampled_videos/RAVDESS/
ln -sf /home/ubuntu/datasets/crema_d_videos/* downsampled_videos/CREMA-D-audio-complete/

# Test data access
echo "Testing dataset access..."
ls -la downsampled_videos/RAVDESS | head
ls -la downsampled_videos/CREMA-D-audio-complete | head

echo "===== Ready to run emotion comparison ====="
echo "To run RAVDESS comparison:"
echo "  ./run_emotion_comparison.sh --ravdess --auto"
echo
echo "To run CREMA-D comparison:"
echo "  ./run_emotion_comparison.sh --cremad --auto"
echo
echo "Results will be in ~/emotion_cmp/comparison_results/"
REMOTE_RUN

# Make the run script executable
ssh -i $KEY $EC2 "chmod +x ~/emotion_cmp/run_ec2.sh"

echo "======================================================="
echo "✅ Deployment Complete!"
echo "======================================================="
echo
echo "To verify dataset setup on EC2:"
echo "  ssh -i $KEY $EC2"
echo "  cd ~/emotion_cmp"
echo "  ./run_ec2.sh"
echo
echo "To run comparisons on EC2:"
echo "  ssh -i $KEY $EC2"
echo "  cd ~/emotion_cmp"
echo "  ./run_emotion_comparison.sh --ravdess --auto     # For RAVDESS"
echo "  ./run_emotion_comparison.sh --cremad --auto      # For CREMA-D"
echo
echo "To download results after completion:"
echo "  rsync -avP -e \"ssh -i $KEY\" $EC2:$REMOTE_DIR/comparison_results/ ./comparison_results_ec2/"
echo
