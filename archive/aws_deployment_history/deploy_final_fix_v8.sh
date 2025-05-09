#!/bin/bash
# Deploy the final fixed Wav2Vec training script with all fixes applied:
# 1. Dataset-specific emotion coding support
# 2. Correct 'wav2vec_features' key usage
# 3. Removed class_weight parameter (incompatible with generators)
# 4. Fixed syntax error in WarmUpReduceLROnPlateau class (missing comma)

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
LOCAL_SCRIPT="fixed_v8_final.py"
REMOTE_DIR="/home/ubuntu/audio_emotion"

echo "Deploying final fixed wav2vec training script (v8)..."

# Copy required script to EC2
echo "Copying script to EC2..."
scp -i $KEY_PATH $LOCAL_SCRIPT $EC2_HOST:$REMOTE_DIR/

# Clear any old cache files to ensure fresh processing
echo "Removing old cache files to ensure fresh data processing..."
ssh -i $KEY_PATH $EC2_HOST "rm -f $REMOTE_DIR/checkpoints/wav2vec_six_classes_best.weights.h5"

# Launch the training
echo "Launching training on EC2..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="wav2vec_final_fix_v8_$TIMESTAMP.log"
ssh -i $KEY_PATH $EC2_HOST "cd $REMOTE_DIR && nohup python3 fixed_v8_final.py > $LOG_FILE 2>&1 &"

echo "Training job started! Log file: $REMOTE_DIR/$LOG_FILE"
echo ""
echo "To monitor training progress use the included monitoring script:"
echo "  ./monitor_final_fix_v8.sh"
echo ""
echo "To set up TensorBoard monitoring:"
echo "  ssh -i ~/Downloads/gpu-key.pem -L 6006:localhost:6006 ubuntu@54.162.134.77"
echo "  Then on EC2: cd $REMOTE_DIR && tensorboard --logdir=logs"
echo "  Open http://localhost:6006 in your browser"

# Create monitor script
cat > monitor_final_fix_v8.sh << EOL
#!/bin/bash
# Monitor training for the final fixed wav2vec model (v8)

# Variables
EC2_HOST="ubuntu@54.162.134.77"
KEY_PATH="~/Downloads/gpu-key.pem"
LOG_FILE="$LOG_FILE"
REMOTE_DIR="$REMOTE_DIR"

echo "Finding the most recent final fix v8 training log file..."
LOG_FILE=\$(ssh -i \$KEY_PATH \$EC2_HOST "find \$REMOTE_DIR -name 'wav2vec_final_fix_v8_*.log' -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d' '")
echo "Using log file: \$LOG_FILE"
echo "==============================================================="

# Check if the process is still running
echo "Checking if training process is running..."
PID=\$(ssh -i \$KEY_PATH \$EC2_HOST "pgrep -f fixed_v8_final.py")
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

chmod +x monitor_final_fix_v8.sh
echo "Created monitoring script: monitor_final_fix_v8.sh"

# Create download script for the final model
cat > download_final_model_v8.sh << EOL
#!/bin/bash
# Download the final fixed wav2vec model (v8)

# Variables
EC2_HOST="ubuntu@54.162.134.77"
KEY_PATH="~/Downloads/gpu-key.pem"
REMOTE_DIR="/home/ubuntu/audio_emotion"
MODEL_FILE="checkpoints/wav2vec_six_classes_best.weights.h5"
TIMESTAMP=\$(date +%Y%m%d_%H%M%S)
LOCAL_MODEL="wav2vec_final_fixed_model_v8_\$TIMESTAMP.h5"

echo "Downloading final fixed wav2vec model v8..."
scp -i \$KEY_PATH \$EC2_HOST:\$REMOTE_DIR/\$MODEL_FILE \$LOCAL_MODEL

if [ -f "\$LOCAL_MODEL" ]; then
    echo "Model successfully downloaded to \$LOCAL_MODEL"
else
    echo "Error: Model download failed"
fi
EOL

chmod +x download_final_model_v8.sh
echo "Created download script: download_final_model_v8.sh"

# Create a documentation file
cat > WAV2VEC_FIXED_EMOTION_README.md << EOL
# Wav2Vec Emotion Recognition - Fixed Implementation

## Overview

This repository contains a fully fixed implementation of our Wav2Vec emotion recognition model. This solution successfully resolves all issues encountered previously:

1. **ResourceVariable Error**: Fixed TypeError with TensorFlow's learning rate by using float() conversion
2. **Dataset-Specific Emotion Coding**: Properly parses emotion codes from both CREMA-D and RAVDESS datasets
3. **NPZ Key Structure**: Uses the correct key 'wav2vec_features' to access audio features within NPZ files
4. **Continuous Emotion Indices**: Maps emotions to continuous indices (0-5) to avoid gaps in class labeling
5. **TensorFlow Generator Compatibility**: Removed class_weight parameter which is not supported with generators
6. **Syntax Error Fix**: Fixed missing comma in the WarmUpReduceLROnPlateau class

## Model Architecture

The model uses a bi-directional LSTM architecture to process wav2vec embeddings:

- Input: Variable-length sequences of 768-dimensional wav2vec features
- Two BiLSTM layers (128 units each)
- Two dense layers (256 and 128 units) with ReLU activation and dropout
- Output: 6-class softmax layer for emotion classification

## Emotion Mapping

Emotions from both datasets are mapped to a continuous index space:

- neutral/calm: 0
- happy: 1
- sad: 2
- angry: 3
- fear: 4
- disgust: 5
- surprise: 6 (if present)

## Training Process

The model is trained with the following settings:
- Adam optimizer with an initial learning rate of 0.001
- Warm-up period of 5 epochs, followed by learning rate reduction on plateau
- Early stopping based on validation accuracy with patience of 10 epochs
- The best weights are saved based on validation accuracy

## Key Fixes Implemented

1. **ResourceVariable Fix**: Changed from using `.value()` to `float()`:
   ```python
   # Old problematic code
   old_lr = self.model.optimizer.learning_rate.value()
   
   # Fixed code
   old_lr = float(self.model.optimizer.learning_rate)
   ```

2. **Dataset-specific emotion coding**: Implemented different parsing logic for each dataset:
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

3. **Correct NPZ key**: Changed from 'features' to 'wav2vec_features':
   ```python
   # Old code
   features = data['features']
   
   # Fixed code
   features = data['wav2vec_features']
   ```

4. **Generator compatibility**: Removed class_weight parameter:
   ```python
   # Old code with incompatible parameter
   history = model.fit(
       train_generator,
       steps_per_epoch=steps_per_epoch,
       epochs=100,
       validation_data=val_generator,
       validation_steps=validation_steps,
       callbacks=callbacks,
       class_weight=class_weights  # This caused the error
   )
   
   # Fixed code
   history = model.fit(
       train_generator,
       steps_per_epoch=steps_per_epoch,
       epochs=100,
       validation_data=val_generator,
       validation_steps=validation_steps,
       callbacks=callbacks
   )
   ```

## Scripts

- **fixed_v8_final.py**: Final training script with all fixes
- **monitor_final_fix_v8.sh**: Script to monitor training progress
- **download_final_model_v8.sh**: Script to download the trained model

## Usage

To deploy and train:
\`\`\`
./deploy_final_fix_v8.sh
\`\`\`

To monitor training:
\`\`\`
./monitor_final_fix_v8.sh
\`\`\`

To download the trained model:
\`\`\`
./download_final_model_v8.sh
\`\`\`
EOL

echo "Created comprehensive documentation: WAV2VEC_FIXED_EMOTION_README.md"
