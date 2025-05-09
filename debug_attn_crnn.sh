#!/bin/bash
#
# Debug script to directly run the ATTN-CRNN training with full error reporting
#

set -euo pipefail

# Configuration
IP="54.162.134.77"
PEM="$HOME/Downloads/gpu-key.pem"

echo "=== DEBUGGING ATTN-CRNN TRAINING ISSUES ==="

ssh -i "$PEM" ubuntu@$IP << 'EOF'
  set -euo pipefail
  
  # Create a debug script on the server with detailed error reporting
  cat > /home/ubuntu/debug_run_attn_crnn.sh << 'DEBUGSCRIPT'
#!/bin/bash
set -eo pipefail

# Setup
cd /home/ubuntu
source /opt/pytorch/bin/activate
export PYTHONPATH=/home/ubuntu/emotion_project:${PYTHONPATH:-}

# Check Python environment
echo "Python version:"
python --version
echo "Python path:"
which python
echo "Verifying TensorFlow and Keras imports..."
python -c "import tensorflow as tf; import tensorflow.keras as keras; print(f'TensorFlow version: {tf.__version__}; Keras version: {tf.keras.__version__}')"

# Verify script exists
if [ -f "/home/ubuntu/emotion_project/scripts/train_attn_crnn.py" ]; then
  echo "✓ Training script exists"
  ls -la /home/ubuntu/emotion_project/scripts/train_attn_crnn.py
else
  echo "✗ Training script NOT found"
  echo "Checking for script in other locations..."
  find /home/ubuntu -name "train_attn_crnn.py" | xargs ls -la
fi

# Verify data access
echo "Checking feature files..."
FEATURE_COUNT=$(find -L /home/ubuntu/emotion_project/wav2vec_features -name "*.npz" | wc -l)
echo "Found $FEATURE_COUNT feature files"
find -L /home/ubuntu/emotion_project/wav2vec_features -name "*.npz" | head -n 5

# Try to run the script directly with all error output
echo "Running training script directly with full error output..."
python /home/ubuntu/emotion_project/scripts/train_attn_crnn.py \
  --data_dirs=/home/ubuntu/emotion_project/wav2vec_features \
  --epochs=50 \
  --batch_size=32 \
  --lr=0.001 \
  --model_save_path=/home/ubuntu/emotion_project/best_attn_crnn_model.h5
DEBUGSCRIPT

  chmod +x /home/ubuntu/debug_run_attn_crnn.sh
  
  # Run the debug script and capture all output
  echo "=== RUNNING DEBUG SCRIPT ==="
  /home/ubuntu/debug_run_attn_crnn.sh || echo "Script failed with error code $?"
EOF

echo "=== DEBUG COMPLETE ==="
