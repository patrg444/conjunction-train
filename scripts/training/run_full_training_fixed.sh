#!/bin/bash
# Script to run full-scale Facenet training (Fixed Version)

set -e

# Ensure execution from script's directory
cd "$(dirname "$0")"

VENV_PATH="$HOME/facenet-venv"
PYTHON_EXEC="$VENV_PATH/bin/python"
PIP_EXEC="$VENV_PATH/bin/pip"

# Set up Python environment if needed
if [ ! -d "$VENV_PATH" ]; then
  echo "Setting up Python environment at $VENV_PATH..."
  python3 -m venv "$VENV_PATH"
  echo "Upgrading pip..."
  "$PIP_EXEC" install --upgrade pip
  # Install dependencies right after creating the venv
  echo "Installing dependencies..."
  "$PIP_EXEC" install tensorflow numpy matplotlib tqdm tensorboard pandas h5py
fi

# Activate environment (though direct execution below is preferred)
# source "$VENV_PATH/bin/activate" # Not strictly needed if executing directly

# Ensure dependencies are installed (in case venv existed but was incomplete)
echo "Ensuring dependencies are installed/updated..."
"$PIP_EXEC" install --upgrade pip
"$PIP_EXEC" install tensorflow numpy matplotlib tqdm tensorboard pandas h5py

# Create directories
mkdir -p models/
mkdir -p logs/

# Define defaults if variables are not set externally
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-100}"

# Run training using the virtual environment's Python
echo "Starting full-scale training with Batch Size: $BATCH_SIZE, Epochs: $EPOCHS..."
"$PYTHON_EXEC" train_facenet_full.py --batch-size "$BATCH_SIZE" --epochs "$EPOCHS"

echo "Training script finished!"
