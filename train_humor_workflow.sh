#!/bin/bash
# This script orchestrates the complete workflow for humor dataset download,
# processing, and training with UR-FUNNY and additional datasets.

set -e  # Exit on error

echo "======================================================================================"
echo "🤣 HUMOR DATASET WORKFLOW: Download, Process, and Train"
echo "======================================================================================"

# Check for required Python packages
echo "Checking and installing required packages..."
pip install pandas numpy tqdm requests gitpython

# Create required directories
mkdir -p datasets/humor_datasets
mkdir -p datasets/manifests/humor

# Step 1: Download and process UR-FUNNY V2 dataset
echo -e "\n1️⃣ Downloading and processing UR-FUNNY V2 dataset"
echo "-------------------------------------------------------------------"
python download_ur_funny_v2.py

# Check if we succeeded
if [ ! -f "datasets/manifests/humor/ur_funny_train_humor.csv" ]; then
  echo "❌ Failed to download UR-FUNNY V2 dataset. Aborting."
  exit 1
fi

echo "✅ UR-FUNNY V2 dataset processed successfully!"

# Step 2: (Optional) Download and process Short-Humor dataset
read -p "Do you want to download and process the Short-Humor dataset as well? (y/n): " download_short

if [[ $download_short == "y" || $download_short == "Y" ]]; then
  echo -e "\n2️⃣ Downloading and processing Short-Humor dataset"
  echo "-------------------------------------------------------------------"
  
  # Try to handle the NLTK dependency
  echo "Installing NLTK dependency..."
  pip install nltk
  
  python download_short_humor.py
  
  # Check if we succeeded
  if [ ! -f "datasets/manifests/humor/short_humor_train_humor.csv" ]; then
    echo "⚠️ Warning: Failed to download Short-Humor dataset. Continuing with just UR-FUNNY."
  else
    echo "✅ Short-Humor dataset processed successfully!"
  fi
else
  echo "Skipping Short-Humor dataset download."
fi

# Step 3: Merge the manifests
echo -e "\n3️⃣ Merging humor manifests"
echo "-------------------------------------------------------------------"
python scripts/merge_humor_manifests.py

# Check if we succeeded
if [ ! -f "datasets/manifests/humor/combined_train_humor.csv" ]; then
  echo "❌ Failed to merge manifests. Aborting."
  exit 1
fi

echo "✅ Manifests merged successfully!"

# Step 4: Train the model
echo -e "\n4️⃣ Ready to train the model"
echo "-------------------------------------------------------------------"
echo "To train the DistilBERT model, run:"
echo "  python enhanced_train_distil_humor.py --train_manifest datasets/manifests/humor/combined_train_humor.csv --val_manifest datasets/manifests/humor/combined_val_humor.csv"

# For EC2
if [ -f "train_distilbert_ec2.sh" ]; then
  echo "Or for EC2 training:"
  echo "  ./train_distilbert_ec2.sh datasets/manifests/humor/combined_train_humor.csv datasets/manifests/humor/combined_val_humor.csv"
fi

read -p "Do you want to automatically train the model now? (y/n): " train_now

if [[ $train_now == "y" || $train_now == "Y" ]]; then
  # Check which training script to use
  if [ -f "train_distilbert_ec2.sh" ]; then
    echo "Running EC2 training script..."
    chmod +x train_distilbert_ec2.sh
    ./train_distilbert_ec2.sh datasets/manifests/humor/combined_train_humor.csv datasets/manifests/humor/combined_val_humor.csv
  else
    echo "Running local training script..."
    python enhanced_train_distil_humor.py --train_manifest datasets/manifests/humor/combined_train_humor.csv --val_manifest datasets/manifests/humor/combined_val_humor.csv
  fi
else
  echo -e "\n🎉 Humor dataset preparation complete! You can now train the model manually using the commands above."
fi

echo -e "\n======================================================================================"
echo "🏁 WORKFLOW COMPLETE"
echo "======================================================================================"
