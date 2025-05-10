#!/bin/bash

# Script to specifically regenerate RAVDESS CNN features to fix the data imbalance

echo "Starting RAVDESS CNN feature extraction..."

# Create output directory
mkdir -p data/ravdess_features_cnn_fixed

# Modify preprocess_cnn_audio_features.py to use the correct paths
TEMP_SCRIPT="scripts/temp_extract_ravdess.py"
cp scripts/preprocess_cnn_audio_features.py $TEMP_SCRIPT

# Update the paths in the temporary script
sed -i.bak "s|RAVDESS_SPEC_INPUT_DIR = os.path.join(project_root, \"data\", \"ravdess_features_spectrogram\")|RAVDESS_SPEC_INPUT_DIR = os.path.join(project_root, \"data\", \"ravdess_features_spectrogram\")|" $TEMP_SCRIPT
sed -i.bak "s|CREMA_D_SPEC_INPUT_DIR = os.path.join(project_root, \"data\", \"crema_d_features_spectrogram\")|CREMA_D_SPEC_INPUT_DIR = \"\"|" $TEMP_SCRIPT
sed -i.bak "s|RAVDESS_CNN_OUTPUT_DIR = os.path.join(project_root, \"data\", \"ravdess_features_cnn_audio\")|RAVDESS_CNN_OUTPUT_DIR = os.path.join(project_root, \"data\", \"ravdess_features_cnn_fixed\")|" $TEMP_SCRIPT
sed -i.bak "s|CREMA_D_CNN_OUTPUT_DIR = os.path.join(project_root, \"data\", \"crema_d_features_cnn_audio\")|CREMA_D_CNN_OUTPUT_DIR = \"\"|" $TEMP_SCRIPT

# Run the modified script
echo "Running feature extraction (RAVDESS-only)..."
python $TEMP_SCRIPT

# Clean up
rm $TEMP_SCRIPT $TEMP_SCRIPT.bak
echo "RAVDESS feature extraction complete!"

# Check results
echo "Verifying extracted features..."
echo "RAVDESS CNN features count: $(find data/ravdess_features_cnn_fixed -type f | wc -l)"
