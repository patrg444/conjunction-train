#!/bin/bash
# Deploy the fixed Wav2Vec training script (fixed_v6_script_final.py) with all fixes applied:
# 1. Dataset-specific emotion coding support
# 2. Correct 'wav2vec_features' key usage
# 3. Fixed missing comma in the set_value() method
# 4. Removed incompatible class_weight parameter

# Check if key exists
if [ ! -f ~/Downloads/gpu-key.pem ]; then
  echo "Error: SSH key not found at ~/Downloads/gpu-key.pem"
  exit 1
fi

# Set permissions if needed
chmod 400 ~/Downloads/gpu-key.pem

# Define variables
EC2_HOST="ubuntu@54.162.134.77"
KEY_PATH="~/Downloads/gpu-key.pem"
LOCAL_SCRIPT="fixed_v6_script_final.py"
REMOTE_DIR="/home/ubuntu/audio_emotion"

echo "Deploying fixed wav2vec training script..."

# Copy required script to EC2
echo "Copying script to EC2..."
scp -i $KEY_PATH $LOCAL_SCRIPT $EC2_HOST:$REMOTE_DIR/

# Clear any old cache files to ensure fresh processing
echo "Removing old cache files to ensure fresh data processing..."
ssh -i $KEY_PATH $EC2_HOST "rm -f $REMOTE_DIR/checkpoints/wav2vec_six_classes_best.weights.h5"

# Launch the training
echo "Launching training on EC2..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="wav2vec_final_fix_$TIMESTAMP.log"
ssh -i $KEY_PATH $EC2_HOST "cd $REMOTE_DIR && nohup python3 fixed_v6_script_final.py > $LOG_FILE 2>&1 &"

echo "Training job started! Log file: $REMOTE_DIR/$LOG_FILE"
echo ""
echo "To monitor training progress use the included monitoring script:"
echo "  ./monitor_final_fix.sh"
echo ""
echo "To set up TensorBoard monitoring:"
echo "  ssh -i ~/Downloads/gpu-key.pem -L 6006:localhost:6006 ubuntu@54.162.134.77"
echo "  Then on EC2: cd $REMOTE_DIR && tensorboard --logdir=logs"
echo "  Open http://localhost:6006 in your browser"

# Create monitor script
cat > monitor_final_fix.sh << EOL
#!/bin/bash
# Monitor training for the fixed wav2vec model

# Variables
EC2_HOST="ubuntu@54.162.134.77"
KEY_PATH="~/Downloads/gpu-key.pem"
LOG_FILE="$LOG_FILE"
REMOTE_DIR="$REMOTE_DIR"

echo "Finding the most recent final fix training log file..."
LOG_FILE=\$(ssh -i \$KEY_PATH \$EC2_HOST "find \$REMOTE_DIR -name 'wav2vec_final_fix_*.log' -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d' '")
echo "Using log file: \$LOG_FILE"
echo "==============================================================="

# Check if the process is still running
echo "Checking if training process is running..."
PID=\$(ssh -i \$KEY_PATH \$EC2_HOST "pgrep -f fixed_v6_script_final.py")
if [ -z "\$PID" ]; then
    echo "PROCESS NOT RUNNING!"
else
    echo "Process is running with PID \$PID"
fi

echo ""
echo "Latest log entries:"
echo "==============================================================="
ssh -i \$KEY_PATH \$EC2_HOST "tail -n 50 \$LOG_FILE"

# Check for emotion distribution information
echo ""
echo "Check for emotion distribution information:"
echo "==============================================================="
ssh -i \$KEY_PATH \$EC2_HOST "grep -A20 'Emotion distribution in dataset' \$LOG_FILE | tail -20"

# Check for proper class encoding
echo ""
echo "Check for proper class encoding:"
echo "==============================================================="
ssh -i \$KEY_PATH \$EC2_HOST "grep 'Number of classes after encoding' \$LOG_FILE"
ssh -i \$KEY_PATH \$EC2_HOST "grep 'Original unique label values' \$LOG_FILE"

# Check for training progress
echo ""
echo "Check for latest training epoch:"
echo "==============================================================="
ssh -i \$KEY_PATH \$EC2_HOST "grep -E 'Epoch [0-9]+/100' \$LOG_FILE | tail -5"
ssh -i \$KEY_PATH \$EC2_HOST "grep -E 'val_accuracy: [0-9.]+' \$LOG_FILE | tail -5"

echo ""
echo "Monitor complete. Run this script again to see updated progress."
EOL

chmod +x monitor_final_fix.sh
echo "Created monitoring script: monitor_final_fix.sh"

# Create download script for the final model
cat > download_final_model.sh << EOL
#!/bin/bash
# Download the fixed wav2vec model

# Variables
EC2_HOST="ubuntu@54.162.134.77"
KEY_PATH="~/Downloads/gpu-key.pem"
REMOTE_DIR="/home/ubuntu/audio_emotion"
MODEL_FILE="checkpoints/wav2vec_six_classes_best.weights.h5"
TIMESTAMP=\$(date +%Y%m%d_%H%M%S)
LOCAL_MODEL="wav2vec_final_fixed_model_\$TIMESTAMP.h5"

echo "Downloading final fixed wav2vec model..."
scp -i \$KEY_PATH \$EC2_HOST:\$REMOTE_DIR/\$MODEL_FILE \$LOCAL_MODEL

if [ -f "\$LOCAL_MODEL" ]; then
    echo "Model successfully downloaded to \$LOCAL_MODEL"
else
    echo "Error: Model download failed"
fi
EOL

chmod +x download_final_model.sh
echo "Created download script: download_final_model.sh"

# Create a documentation file
cat > WAV2VEC_CONSOLIDATED_FIX_README.md << EOL
# Wav2Vec Emotion Recognition - Consolidated Fix

## Overview

This repository contains our fixed implementation of the Wav2Vec emotion recognition model. 
All issues have been consolidated and fixed in one script rather than creating multiple versions:

1. **ResourceVariable Error**: Fixed TypeError with TensorFlow's learning rate by using float() conversion
2. **Dataset-Specific Emotion Coding**: Properly parses emotion codes from both CREMA-D and RAVDESS datasets
3. **NPZ Key Structure**: Uses the correct key 'wav2vec_features' to access audio features within NPZ files
4. **Continuous Emotion Indices**: Maps emotions to continuous indices (0-6) to avoid gaps in class labeling
5. **TensorFlow Generator Compatibility**: Removed class_weight parameter which is not supported with generators
6. **Syntax Fix**: Added missing comma in the WarmUpReduceLROnPlateau class

## Model Architecture

The model uses a bi-directional LSTM architecture to process wav2vec embeddings:

- Input: Variable-length sequences of 768-dimensional wav2vec features
- Two BiLSTM layers (128 units each)
- Two dense layers (256 and 128 units) with ReLU activation and dropout
- Output: Softmax layer for emotion classification (6-7 classes depending on dataset)

## Emotion Mapping

Emotions from both datasets are mapped to a continuous index space:

- neutral/calm: 0
- happy: 1
- sad: 2
- angry: 3
- fear: 4
- disgust: 5
- surprise: 6 (if present)

## Key Fixes Implemented

1. **ResourceVariable Fix**: Changed from using `.value()` to `float()`:
   ```python
   old_lr = float(self.model.optimizer.learning_rate)
   ```

2. **Syntax Error Fix**: Added missing comma in method call:
   ```python
   # Original (with error)
   tf.keras.backend.set_value(self.model.optimizer.learning_rate warmup_lr)
   
   # Fixed
   tf.keras.backend.set_value(self.model.optimizer.learning_rate, warmup_lr)
   ```

3. **Generator Compatibility**: Removed incompatible class_weight parameter:
   ```python
   # Fixed code - removed class_weight parameter
   history = model.fit(
       train_generator,
       steps_per_epoch=steps_per_epoch,
       epochs=100,
       validation_data=val_generator,
       validation_steps=validation_steps,
       callbacks=callbacks
   )
   ```

4. **Dataset-specific Parsing**: Different logic for different datasets:
   ```python
   # CREMA-D parsing
   if base_name.startswith('cremad_'):
       parts = base_name.split('_')
       emotion_code = parts[3]  # ANG, DIS, FEA, etc.
       emotion = cremad_code_to_emotion.get(emotion_code)
   
   # RAVDESS parsing
   elif base_name.startswith('ravdess_'):
       parts = base_name[8:].split('-')
       emotion_code = parts[2]  # Third digit is emotion code
       emotion = ravdess_code_to_emotion.get(emotion_code)
   ```

## Scripts

- **fixed_v6_script_final.py**: Training script with all fixes consolidated
- **monitor_final_fix.sh**: Script to monitor training progress
- **download_final_model.sh**: Script to download the trained model

## Usage

To deploy and train:
\`\`\`
./deploy_final_fix_revised.sh
\`\`\`

To monitor training:
\`\`\`
./monitor_final_fix.sh
\`\`\`

To download the trained model:
\`\`\`
./download_final_model.sh
\`\`\`
EOL

echo "Created documentation: WAV2VEC_CONSOLIDATED_FIX_README.md"
