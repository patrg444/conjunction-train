#!/bin/bash
# Script to prepare data and code for AWS training

# Create a directory for the AWS setup
mkdir -p aws-temp

# Copy required Python scripts
echo "Copying required Python scripts..."
cp ../scripts/train_branched_6class.py aws-temp/
cp ../scripts/sequence_data_generator.py aws-temp/
cp ../scripts/train_branched_dynamic_funcs.py aws-temp/
cp ../requirements.txt aws-temp/

# Create a compressed archive for fast transfer
echo "Creating an archive of the feature directories..."
tar -czf aws-temp/ravdess_features_facenet.tar.gz -C .. ravdess_features_facenet/
tar -czf aws-temp/crema_d_features_facenet.tar.gz -C .. crema_d_features_facenet/

echo "Creating model output directory archive (if it exists)..."
if [ -d "../models/branched_6class" ]; then
    tar -czf aws-temp/existing_models.tar.gz -C .. models/branched_6class/
fi

echo "Preparation complete. Files are ready in the aws-temp directory."
