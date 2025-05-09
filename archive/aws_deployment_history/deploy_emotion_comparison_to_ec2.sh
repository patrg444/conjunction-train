#!/bin/bash
# Deployment script to push the emotion comparison framework to EC2 and run it
set -e

# Configure paths and credentials
KEY=~/Downloads/gpu-key.pem
EC2=ubuntu@54.162.134.77
SRC=$(pwd)  # Current directory - emotion recognition project root
REMOTE_DIR=/home/ubuntu/emotion_cmp

# Make sure key has correct permissions
chmod 400 $KEY

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

# Step 2: Sync code (excluding large dirs and temp files)
echo "📤 Syncing code to EC2..."
rsync -avP --exclude '.git' \
      --exclude '__pycache__' \
      --exclude '*.pyc' \
      --exclude '*.npz' \
      --exclude '*_features.tar.gz' \
      --exclude '*_features_facenet.tar.gz' \
      --exclude 'crema_d_*.tar.gz' \
      --exclude 'downsampled_videos/' \
      --exclude 'comparison_results' \
      -e "ssh -i $KEY" \
      "$SRC/emotion_comparison/" "$EC2:$REMOTE_DIR/emotion_comparison/"

echo "📤 Syncing shell scripts and requirements..."
rsync -avP -e "ssh -i $KEY" \
      "$SRC/run_emotion_comparison.sh" \
      "$SRC/requirements.txt" \
      "$EC2:$REMOTE_DIR/"

echo "✅ Code sync complete!"

# Step 3: Set up environment on EC2
echo "🛠️ Setting up Python environment on EC2..."
ssh -i $KEY $EC2 <<'REMOTE_SETUP'
set -e
cd ~

echo "    Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3-venv build-essential ffmpeg git python3-dev

echo "    Creating Python virtual environment..."
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install --upgrade pip

echo "    Installing Python dependencies..."
pip install scikit-learn seaborn matplotlib pandas
cd ~/emotion_cmp
pip install -r emotion_comparison/requirements.txt

echo "    Setting up execution permissions..."
chmod +x run_emotion_comparison.sh
REMOTE_SETUP

echo "✅ Environment setup complete!"

# Step 4: Verify datasets are present
echo "🔍 Verifying datasets on EC2..."
ssh -i $KEY $EC2 <<'REMOTE_VERIFY'
if [ -d "/home/ubuntu/datasets/ravdess_videos" ]; then
    echo "✅ RAVDESS dataset found at /home/ubuntu/datasets/ravdess_videos"
    echo "   Sample files:"
    ls -la /home/ubuntu/datasets/ravdess_videos | head -5
    echo "   Total files: $(find /home/ubuntu/datasets/ravdess_videos -type f | wc -l)"
else
    echo "❌ RAVDESS dataset not found at /home/ubuntu/datasets/ravdess_videos"
fi

if [ -d "/home/ubuntu/datasets/crema_d_videos" ]; then
    echo "✅ CREMA-D dataset found at /home/ubuntu/datasets/crema_d_videos"
    echo "   Sample files:"
    ls -la /home/ubuntu/datasets/crema_d_videos | head -5
    echo "   Total files: $(find /home/ubuntu/datasets/crema_d_videos -type f | wc -l)"
else
    echo "❌ CREMA-D dataset not found at /home/ubuntu/datasets/crema_d_videos"
fi
REMOTE_VERIFY

# Step 5: Create run script with correct paths for EC2
echo "📝 Creating run script for EC2..."
ssh -i $KEY $EC2 "cat > ~/emotion_cmp/run_ec2.sh" << 'REMOTE_RUN'
#!/bin/bash
# Run script for the emotion comparison framework on EC2
set -e

# Activate Python environment
source ~/venv/bin/activate
cd ~/emotion_cmp

# Create symbolic links to the datasets
mkdir -p downsampled_videos/RAVDESS
mkdir -p downsampled_videos/CREMA-D-audio-complete
ln -sf /home/ubuntu/datasets/ravdess_videos/* downsampled_videos/RAVDESS/
ln -sf /home/ubuntu/datasets/crema_d_videos/* downsampled_videos/CREMA-D-audio-complete/

# Run RAVDESS comparison with grid search
echo "===== Running RAVDESS Comparison with Grid Search ====="
./run_emotion_comparison.sh --ravdess --grid-search --auto

# Run CREMA-D comparison with grid search
echo "===== Running CREMA-D Comparison with Grid Search ====="
./run_emotion_comparison.sh --cremad --grid-search --auto

# Run cross-dataset analysis
echo "===== Running Cross-Dataset Analysis ====="
python emotion_comparison/cross_dataset_analysis.py \
  --ravdess_dir comparison_results/RAVDESS_results \
  --cremad_dir comparison_results/CREMAD_results \
  --output_dir comparison_results/combined_results

echo "===== All Comparisons Complete ====="
echo "Results are in ~/emotion_cmp/comparison_results/"
REMOTE_RUN

# Make the run script executable
ssh -i $KEY $EC2 "chmod +x ~/emotion_cmp/run_ec2.sh"

echo "======================================================="
echo "✅ Deployment Complete!"
echo "======================================================="
echo
echo "To run the full comparison on EC2:"
echo "  ssh -i $KEY $EC2"
echo "  cd ~/emotion_cmp"
echo "  ./run_ec2.sh"
echo
echo "To download results after completion:"
echo "  rsync -avP -e \"ssh -i $KEY\" $EC2:$REMOTE_DIR/comparison_results/ ./comparison_results_ec2/"
echo "  open ./comparison_results_ec2/combined_results/*.html"
echo
