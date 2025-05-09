#!/bin/bash

# EC2 connection details (using the specific key information)
AWS_IP="54.162.134.77"
SSH_USER="ubuntu"
SSH_HOST="$SSH_USER@$AWS_IP"
SSH_KEY="$HOME/Downloads/gpu-key.pem"

# Check if the key exists
if [[ ! -f "$SSH_KEY" ]]; then
  echo "ERROR: SSH key not found at $SSH_KEY"
  exit 1
fi

# Ensure key has proper permissions
chmod 400 "$SSH_KEY" 2>/dev/null || true

# SSH command
SSH_COMMAND="ssh -i $SSH_KEY $SSH_HOST"
SCP_COMMAND="scp -i $SSH_KEY"

# Remote script to execute on EC2
echo "Creating remote script to download and extract datasets..."

$SSH_COMMAND "cat > /home/ubuntu/download_datasets.sh" << 'REMOTE_SCRIPT'
#!/bin/bash

# Make sure our terminal shows colors for better readability
export TERM=xterm-color

echo -e "\033[1;34m=== EMOTION RECOGNITION DATASET UPLOADER ===\033[0m"
echo -e "\033[1;34m=== Uploading RAVDESS and CREMA-D video datasets to EC2 ===\033[0m"

# Define target directories (these will be our final paths)
RAVDESS_VIDEOS_PATH="/home/ubuntu/datasets/ravdess_videos"
CREMA_D_VIDEOS_PATH="/home/ubuntu/datasets/crema_d_videos"

echo -e "\033[1;36m[1/5] Creating target directories...\033[0m"
mkdir -p "$RAVDESS_VIDEOS_PATH"
mkdir -p "$CREMA_D_VIDEOS_PATH"
sudo chown -R ubuntu:ubuntu /home/ubuntu/datasets 2>/dev/null || true

echo -e "\033[1;36m[2/5] Downloading datasets (this may take some time)...\033[0m"
cd /home/ubuntu/datasets

# RAVDESS dataset - Speech+Song video zip (≈ 2.8 GB)
echo -e "\033[1;33m   >>> Downloading RAVDESS video dataset...\033[0m"
wget -q --show-progress https://zenodo.org/record/1188976/files/Video_Speech_Actor_01-24.zip -O ravdess.zip

# CREMA-D dataset full video zip (≈ 15 GB)
echo -e "\033[1;33m   >>> Downloading CREMA-D video dataset...\033[0m"
wget -q --show-progress https://zenodo.org/record/1225427/files/CREMA-D.zip -O crema_d.zip

echo -e "\033[1;36m[3/5] Extracting datasets...\033[0m"

# RAVDESS: Unpack Actor_* folders
echo -e "\033[1;33m   >>> Extracting RAVDESS videos to $RAVDESS_VIDEOS_PATH...\033[0m"
mkdir -p ravdess_tmp
unzip -q ravdess.zip -d ravdess_tmp
mv ravdess_tmp/Video_Speech_Actor_* "$RAVDESS_VIDEOS_PATH/"
# Rename to strip the "Video_Speech_" prefix
for dir in "$RAVDESS_VIDEOS_PATH"/Video_Speech_Actor_*; do
  if [ -d "$dir" ]; then
    new_name="${dir/Video_Speech_/}"
    mv "$dir" "$new_name"
  fi
done
rm -rf ravdess_tmp ravdess.zip

# CREMA-D: Unpack all video files 
echo -e "\033[1;33m   >>> Extracting CREMA-D videos to $CREMA_D_VIDEOS_PATH...\033[0m"
mkdir -p crema_d_tmp
unzip -q crema_d.zip -d crema_d_tmp
# Move all video files, adjusting based on how they're organized in the zip
find crema_d_tmp -type f \( -name "*.mp4" -o -name "*.flv" \) -exec mv {} "$CREMA_D_VIDEOS_PATH/" \;
rm -rf crema_d_tmp crema_d.zip

echo -e "\033[1;36m[4/5] Verifying datasets...\033[0m"

# Count files to verify extraction worked correctly
RAVDESS_COUNT=$(find "$RAVDESS_VIDEOS_PATH" -type f -name '*.mp4' | wc -l)
CREMA_D_COUNT=$(find "$CREMA_D_VIDEOS_PATH" -type f \( -name "*.mp4" -o -name "*.flv" \) | wc -l)

echo -e "\033[1;33m   >>> RAVDESS videos: $RAVDESS_COUNT files\033[0m"
echo -e "\033[1;33m   >>> CREMA-D videos: $CREMA_D_COUNT files\033[0m"

echo -e "\033[1;36m[5/5] Reporting paths for extract_fer_features.py...\033[0m"

echo -e "\033[1;32m=== UPLOAD COMPLETE ===\033[0m"
echo -e "\033[1;32m"
echo "RAVDESS_VIDEOS_PATH=$RAVDESS_VIDEOS_PATH"
echo "CREMA_D_VIDEOS_PATH=$CREMA_D_VIDEOS_PATH"
echo -e "\033[0m"
echo -e "\033[1;34mNext steps: Update these paths in scripts/extract_fer_features.py and run the extraction\033[0m"
REMOTE_SCRIPT

# Make the script executable
$SSH_COMMAND "chmod +x /home/ubuntu/download_datasets.sh"

# Execute the script and capture output
echo "Executing remote script to download and extract datasets..."
echo "This will take some time as it downloads ~18GB of data and extracts it."
echo "--------------------------------------------------------------------"
$SSH_COMMAND "/home/ubuntu/download_datasets.sh"
echo "--------------------------------------------------------------------"

echo "Done! The datasets have been uploaded to the EC2 instance."
