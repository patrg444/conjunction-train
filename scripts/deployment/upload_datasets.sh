#!/bin/bash

# Make sure our terminal shows colors for better readability
export TERM=xterm-color

echo -e "\e[1;34m=== EMOTION RECOGNITION DATASET UPLOADER ===\e[0m"
echo -e "\e[1;34m=== Uploading RAVDESS and CREMA-D video datasets to EC2 ===\e[0m"

# Define target directories (these will be our final paths)
RAVDESS_VIDEOS_PATH="/home/ubuntu/datasets/ravdess_videos"
CREMA_D_VIDEOS_PATH="/home/ubuntu/datasets/crema_d_videos"

echo -e "\e[1;36m[1/5] Creating target directories...\e[0m"
sudo mkdir -p "$RAVDESS_VIDEOS_PATH"
sudo mkdir -p "$CREMA_D_VIDEOS_PATH"
sudo chown -R ubuntu:ubuntu /home/ubuntu/datasets

echo -e "\e[1;36m[2/5] Downloading datasets (this may take some time)...\e[0m"
cd /home/ubuntu/datasets

# RAVDESS dataset - Speech+Song video zip (≈ 2.8 GB)
echo -e "\e[1;33m   >>> Downloading RAVDESS video dataset...\e[0m"
wget -q --show-progress https://zenodo.org/record/1188976/files/Video_Speech_Actor_01-24.zip -O ravdess.zip

# CREMA-D dataset full video zip (≈ 15 GB)
echo -e "\e[1;33m   >>> Downloading CREMA-D video dataset...\e[0m"
wget -q --show-progress https://zenodo.org/record/1225427/files/CREMA-D.zip -O crema_d.zip

echo -e "\e[1;36m[3/5] Extracting datasets...\e[0m"

# RAVDESS: Unpack Actor_* folders
echo -e "\e[1;33m   >>> Extracting RAVDESS videos to $RAVDESS_VIDEOS_PATH...\e[0m"
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
echo -e "\e[1;33m   >>> Extracting CREMA-D videos to $CREMA_D_VIDEOS_PATH...\e[0m"
mkdir -p crema_d_tmp
unzip -q crema_d.zip -d crema_d_tmp
# Move all video files, adjusting based on how they're organized in the zip
find crema_d_tmp -type f \( -name "*.mp4" -o -name "*.flv" \) -exec mv {} "$CREMA_D_VIDEOS_PATH/" \;
rm -rf crema_d_tmp crema_d.zip

echo -e "\e[1;36m[4/5] Verifying datasets...\e[0m"

# Count files to verify extraction worked correctly
RAVDESS_COUNT=$(find "$RAVDESS_VIDEOS_PATH" -type f -name '*.mp4' | wc -l)
CREMA_D_COUNT=$(find "$CREMA_D_VIDEOS_PATH" -type f \( -name "*.mp4" -o -name "*.flv" \) | wc -l)

echo -e "\e[1;33m   >>> RAVDESS videos: $RAVDESS_COUNT files\e[0m"
echo -e "\e[1;33m   >>> CREMA-D videos: $CREMA_D_COUNT files\e[0m"

echo -e "\e[1;36m[5/5] Reporting paths for extract_fer_features.py...\e[0m"

echo -e "\e[1;32m=== UPLOAD COMPLETE ===\e[0m"
echo -e "\e[1;32m"
echo "RAVDESS_VIDEOS_PATH=$RAVDESS_VIDEOS_PATH"
echo "CREMA_D_VIDEOS_PATH=$CREMA_D_VIDEOS_PATH"
echo -e "\e[0m"
echo -e "\e[1;34mNext steps: Update these paths in scripts/extract_fer_features.py and run the extraction\e[0m"

# Clean up
cd - > /dev/null
