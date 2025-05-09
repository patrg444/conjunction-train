#!/usr/bin/env bash
# Script to install dependencies for the emotion recognition feature extraction project

set -e  # Exit on any error

echo "===== Installing dependencies for Emotion Recognition Feature Extraction Comparison ====="

# Create a virtual environment (optional)
if [ "$1" == "--venv" ]; then
    echo "Creating virtual environment..."
    python -m venv emotion_venv
    source emotion_venv/bin/activate
    echo "Virtual environment activated."
fi

# Install pip dependencies from requirements.txt
echo "Installing Python dependencies from requirements.txt..."
pip install -r emotion_comparison/requirements.txt

# Install OpenFace if needed (this is a complex process, simplified here)
if ! command -v FeatureExtraction >/dev/null 2>&1; then
    echo "OpenFace is not installed. It requires manual installation."
    echo "For AU extraction, please follow instructions at https://github.com/TadasBaltrusaitis/OpenFace"
    echo "Skip this step if you don't plan to use the FACS method."
fi

# Verify essential dependencies
echo "Verifying essential dependencies..."
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')"

# Check for GPU availability
echo "Checking for GPU availability..."
python -c "
import tensorflow as tf
print(f'TensorFlow sees {len(tf.config.list_physical_devices(\"GPU\"))} GPU(s)')
import torch
print(f'PyTorch CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'PyTorch CUDA device: {torch.cuda.get_device_name(0)}')
"

echo "===== Installation complete ====="
echo "You can now run the dataset test script with: ./run_dataset_test.sh"
