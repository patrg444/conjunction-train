#!/bin/bash
set -e

# Function to show help message
show_help() {
  cat << EOF
Usage: $(basename "$0") [OPTIONS]

This script uploads trained DeBERTa humor detection models to an EC2 instance.

OPTIONS:
  --help, -h          Show this help message and exit
  --dry-run           Print commands without executing them
  --key PATH          Path to SSH key (default: /Users/patrickgloria/Downloads/gpu-key.pem)
  --host HOST         EC2 instance hostname or IP (using aws_instance_ip.txt by default)
  --model_dir DIR     Local directory containing trained model (default: training_logs_humor)
  --remote_dir DIR    Remote directory on EC2 instance (default: ~/humor_models)
  --include_manifests Also upload humor manifests and datasets
  --include_source    Also upload source code files

Examples:
  $(basename "$0")                           # Upload with default options
  $(basename "$0") --dry-run                 # Show commands without executing
  $(basename "$0") --include_manifests       # Upload model with manifests
EOF
  exit 0
}

# Default values for options
DRY_RUN=0
KEY_PATH="/Users/patrickgloria/Downloads/gpu-key.pem"
HOST_FILE="aws_instance_ip.txt"
LOCAL_MODEL_DIR="training_logs_humor"
REMOTE_DIR="~/humor_models"
INCLUDE_MANIFESTS=0
INCLUDE_SOURCE=0

# Process arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      show_help
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --key)
      KEY_PATH="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --model_dir)
      LOCAL_MODEL_DIR="$2"
      shift 2
      ;;
    --remote_dir)
      REMOTE_DIR="$2"
      shift 2
      ;;
    --include_manifests)
      INCLUDE_MANIFESTS=1
      shift
      ;;
    --include_source)
      INCLUDE_SOURCE=1
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# If HOST is not explicitly set, read from HOST_FILE
if [[ -z "$HOST" ]]; then
  if [[ -f "$HOST_FILE" ]]; then
    HOST=$(cat "$HOST_FILE")
  else
    echo "Error: AWS instance IP file ($HOST_FILE) not found and no host specified."
    echo "Either create this file or use --host option."
    exit 1
  fi
fi

# Generate a timestamp for the remote directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REMOTE_MODEL_DIR="${REMOTE_DIR}/deberta_humor_${TIMESTAMP}"

# Function to run or print commands
run_cmd() {
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "DRY RUN: $1"
  else
    echo "Executing: $1"
    eval "$1"
  fi
}

# Create remote directory
run_cmd "ssh -i \"$KEY_PATH\" ubuntu@$HOST \"mkdir -p $REMOTE_MODEL_DIR\""

# Upload model files
run_cmd "rsync -av --progress -e \"ssh -i $KEY_PATH\" \"$LOCAL_MODEL_DIR/\" ubuntu@$HOST:\"$REMOTE_MODEL_DIR/model/\""

# Upload source files
if [[ $INCLUDE_SOURCE -eq 1 ]]; then
  run_cmd "rsync -av --progress -e \"ssh -i $KEY_PATH\" train_deberta_humor.py ubuntu@$HOST:\"$REMOTE_MODEL_DIR/\""
  run_cmd "rsync -av --progress -e \"ssh -i $KEY_PATH\" shell/train_deberta_humor.sh ubuntu@$HOST:\"$REMOTE_MODEL_DIR/\""
fi

# Upload manifests and datasets if requested
if [[ $INCLUDE_MANIFESTS -eq 1 ]]; then
  run_cmd "rsync -av --progress -e \"ssh -i $KEY_PATH\" datasets/manifests/humor/ ubuntu@$HOST:\"$REMOTE_MODEL_DIR/manifests/\""
fi

# Generate a README file with usage instructions
cat << EOF > /tmp/model_readme.md
# DeBERTa Humor Detection Model

This model was trained with DeBERTa for humor text classification and uploaded on $(date).

## Contents

- \`model/\` - Trained model files and checkpoints
- \`manifests/\` - Dataset manifest files (if included)

## Usage

To run inference with this model:

\`\`\`python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the model and tokenizer
model_path = "$REMOTE_MODEL_DIR/model"
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Run inference
text = "Your text to classify"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
with torch.no_grad():
    outputs = model(**inputs)
    
probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
prediction = torch.argmax(probabilities, dim=-1).item()
probability = probabilities[0][prediction].item()

print(f"Prediction: {'Humorous' if prediction == 1 else 'Not humorous'}")
print(f"Confidence: {probability:.2f}")
\`\`\`

For batch inference or to continue training, see the full documentation.
EOF

run_cmd "scp -i \"$KEY_PATH\" /tmp/model_readme.md ubuntu@$HOST:\"$REMOTE_MODEL_DIR/README.md\""

if [[ $DRY_RUN -eq 0 ]]; then
  echo "Deployment completed successfully!"
  echo "Model deployed to: $HOST:$REMOTE_MODEL_DIR"
else
  echo "Dry run completed. No files were actually transferred."
fi
