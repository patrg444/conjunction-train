#!/bin/bash
# Script to set up the AWS instance environment

# Update package listings
echo "Updating package listings..."
sudo apt-get update

# Install required system packages
echo "Installing required system packages..."
sudo apt-get install -y python3-pip python3-dev unzip

# Create directories
echo "Creating project directories..."
mkdir -p ravdess_features_facenet
mkdir -p crema_d_features_facenet
mkdir -p models/branched_6class

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install tensorflow-gpu  # Use GPU-optimized TensorFlow
pip install -r requirements.txt

# Extract feature data
echo "Extracting feature data..."
tar -xzf ravdess_features_facenet.tar.gz
tar -xzf crema_d_features_facenet.tar.gz

# Extract existing models if available
if [ -f "existing_models.tar.gz" ]; then
    echo "Extracting existing models..."
    tar -xzf existing_models.tar.gz
fi

echo "Setup complete. Environment is ready for training."
