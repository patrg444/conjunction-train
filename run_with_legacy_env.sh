#!/bin/bash
# Load the H5 model using the legacy environment with TensorFlow 1.15

# ANSI colors for better output
BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}Loading H5 Model with Legacy Environment${NC}"
echo -e "${BLUE}=============================================${NC}"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda is not installed or not in PATH${NC}"
    echo -e "${YELLOW}Please install Miniconda or Anaconda first:${NC}"
    echo -e "${YELLOW}https://docs.conda.io/en/latest/miniconda.html${NC}"
    echo -e "${YELLOW}Or run setup_legacy_environment.sh to set up the environment${NC}"
    exit 1
fi

# Environment name
ENV_NAME="emotion_recognition_legacy"

# Check if environment exists
if ! conda info --envs | grep -q $ENV_NAME; then
    echo -e "${YELLOW}Environment '$ENV_NAME' does not exist.${NC}"
    echo -e "${YELLOW}Running setup_legacy_environment.sh to create it...${NC}"
    
    # Check if setup script exists
    if [ ! -f "./setup_legacy_environment.sh" ]; then
        echo -e "${RED}Error: setup_legacy_environment.sh not found${NC}"
        exit 1
    fi
    
    # Make it executable
    chmod +x ./setup_legacy_environment.sh
    
    # Run the setup script
    ./setup_legacy_environment.sh
    
    # Check if setup was successful
    if ! conda info --envs | grep -q $ENV_NAME; then
        echo -e "${RED}Failed to create environment. Exiting.${NC}"
        exit 1
    fi
fi

# Make sure model file exists
MODEL_PATH="models/dynamic_padding_no_leakage/model_best.h5"
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model file not found at $MODEL_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}Using conda environment: $ENV_NAME${NC}"
echo -e "${GREEN}Loading model: $MODEL_PATH${NC}"

# Create a temporary script file to load the model
TEMP_SCRIPT="temp_model_loader.py"

cat > $TEMP_SCRIPT << 'EOF'
#!/usr/bin/env python
"""
Load H5 Model with TensorFlow 1.15

This script loads the pre-trained h5 model from models/dynamic_padding_no_leakage/model_best.h5
using TensorFlow 1.15 which supports the time_major parameter.
This model is from the train_branched_no_leakage.py architecture.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Bidirectional, Concatenate
from tensorflow.keras.layers import TimeDistributed, Conv1D, MaxPooling1D, Flatten, BatchNormalization, Masking

# Display TensorFlow version for verification
print(f"TensorFlow version: {tf.__version__}")

def create_model_architecture(audio_feature_dim=88, video_feature_dim=512, num_classes=6):
    """
    Recreate the original model architecture from train_branched_no_leakage.py
    """
    print(f"Creating model architecture matching train_branched_no_leakage.py")
    print(f"- Audio feature dimension: {audio_feature_dim}")
    print(f"- Video feature dimension: {video_feature_dim}")
    
    # Audio branch with masking
    audio_input = Input(shape=(None, audio_feature_dim), name='audio_input')
    
    # Add masking layer to handle padding
    audio_masked = Masking(mask_value=0.0)(audio_input)
    
    # Apply 1D convolutions to extract local patterns
    audio_x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(audio_masked)
    audio_x = BatchNormalization()(audio_x)
    audio_x = MaxPooling1D(pool_size=2)(audio_x)
    
    audio_x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(audio_x)
    audio_x = BatchNormalization()(audio_x)
    audio_x = MaxPooling1D(pool_size=2)(audio_x)
    
    # Apply bidirectional LSTM for temporal features
    audio_x = Bidirectional(LSTM(128, return_sequences=True, time_major=False))(audio_x)
    audio_x = Dropout(0.3)(audio_x)
    audio_x = Bidirectional(LSTM(64))(audio_x)
    audio_x = Dense(128, activation='relu')(audio_x)
    audio_x = Dropout(0.4)(audio_x)
    
    # Video branch with masking
    video_input = Input(shape=(None, video_feature_dim), name='video_input')
    
    # Add masking layer to handle padding
    video_masked = Masking(mask_value=0.0)(video_input)
    
    # FaceNet features already have high dimensionality, so we'll use LSTM directly
    video_x = Bidirectional(LSTM(256, return_sequences=True, time_major=False))(video_masked)
    video_x = Dropout(0.3)(video_x)
    video_x = Bidirectional(LSTM(128))(video_x)
    video_x = Dense(256, activation='relu')(video_x)
    video_x = Dropout(0.4)(video_x)
    
    # Merge branches with more sophisticated fusion
    merged = Concatenate()([audio_x, video_x])
    merged = Dense(256, activation='relu')(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(0.5)(merged)
    merged = Dense(128, activation='relu')(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(0.4)(merged)
    
    # Output layer
    output = Dense(num_classes, activation='softmax')(merged)
    
    # Create model
    model = Model(inputs=[video_input, audio_input], outputs=output)
    
    return model

def load_trained_model():
    """
    Load the pre-trained model directly from h5 file
    """
    model_path = os.path.join('models', 'dynamic_padding_no_leakage', 'model_best.h5')
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    
    print(f"Loading model from: {model_path}")
    
    try:
        # Method 1: Try to load model directly
        print("Trying direct model loading...")
        model = load_model(model_path)
        print("Model loaded successfully with direct loading!")
        return model
    except Exception as e:
        print(f"Error with direct loading: {str(e)}")
        
        try:
            # Method 2: Create the architecture and load weights
            print("Creating model architecture and loading weights...")
            model = create_model_architecture()
            model.load_weights(model_path)
            print("Weights loaded successfully!")
            return model
        except Exception as e2:
            print(f"Error loading weights: {str(e2)}")
            return None

def test_model(model):
    """
    Test the loaded model with dummy inputs
    """
    if model is None:
        return
    
    # Get input shapes from model
    input_shapes = [(input.name, input.shape) for input in model.inputs]
    print(f"Model input shapes: {input_shapes}")
    
    # Create some dummy input data
    dummy_inputs = []
    for _, shape in input_shapes:
        # Create a batch of 1 with dynamic sequence length of 10
        # Replace None with a concrete value (10) for testing
        concrete_shape = [1] + [10 if dim is None else dim for dim in shape[1:]]
        dummy_inputs.append(np.random.random(concrete_shape))
    
    # Run prediction
    try:
        print("Testing model with dummy inputs...")
        predictions = model.predict(dummy_inputs)
        print(f"Prediction shape: {predictions.shape}")
        print(f"Prediction: {predictions}")
        print("Model test successful!")
    except Exception as e:
        print(f"Error testing model: {str(e)}")

if __name__ == "__main__":
    print("Starting model loading script with TensorFlow 1.15...")
    model = load_trained_model()
    if model is not None:
        test_model(model)
    else:
        print("Model loading failed, skipping test.")
EOF

# Make it executable
chmod +x $TEMP_SCRIPT

# Activate the environment and run the script
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Run the model loading script
python $TEMP_SCRIPT

# Capture status
STATUS=$?

# Deactivate the environment
conda deactivate

# Clean up
rm $TEMP_SCRIPT

# Check status
if [ $STATUS -eq 0 ]; then
    echo -e "${GREEN}Model loaded successfully with legacy environment!${NC}"
else
    echo -e "${RED}Failed to load model with legacy environment.${NC}"
fi

echo -e "${BLUE}=============================================${NC}"
