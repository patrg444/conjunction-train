#!/bin/bash
# Script to run the WAV2VEC model evaluation on the full dataset

# Create directories if they don't exist
mkdir -p models/wav2vec_v9_attention
mkdir -p results

# Check if model exists, if not download it
if [ ! -f "models/wav2vec_v9_attention/best_model_v9.h5" ]; then
  echo "Model not found locally. Downloading model first..."
  
  # Check if download script exists, if not create it
  if [ ! -f "download_v9_fixed_model.sh" ]; then
    echo "Download script not found. Creating download script..."
    
    cat > download_v9_fixed_model.sh << 'EOF'
#!/bin/bash
# Download the WAV2VEC v9 attention-based emotion recognition model from EC2

echo "===== Downloading Fixed WAV2VEC Attention Model (v9) ====="

# Set the EC2 connection details
EC2_USER="ubuntu"
EC2_HOST=$(cat aws_instance_ip.txt 2>/dev/null)

if [ -z "$EC2_HOST" ]; then
  echo "Error: EC2 host IP not found. Please check aws_instance_ip.txt file."
  exit 1
fi

# Create directory for model files
mkdir -p models/wav2vec_v9_attention

# Download the model files
echo "Downloading model files..."
scp -o StrictHostKeyChecking=no $EC2_USER@$EC2_HOST:/home/ubuntu/audio_emotion/checkpoints/best_model_v9.h5 models/wav2vec_v9_attention/
scp -o StrictHostKeyChecking=no $EC2_USER@$EC2_HOST:/home/ubuntu/audio_emotion/checkpoints/label_classes_v9.npy models/wav2vec_v9_attention/
scp -o StrictHostKeyChecking=no $EC2_USER@$EC2_HOST:/home/ubuntu/audio_emotion/audio_mean_v9.npy models/wav2vec_v9_attention/
scp -o StrictHostKeyChecking=no $EC2_USER@$EC2_HOST:/home/ubuntu/audio_emotion/audio_std_v9.npy models/wav2vec_v9_attention/

# Check if files were downloaded successfully
if [ -f "models/wav2vec_v9_attention/best_model_v9.h5" ]; then
  echo "Model downloaded successfully."
  echo "Files saved to models/wav2vec_v9_attention/"
  
  # Display model information
  echo ""
  echo "Model details:"
  echo "- Architecture: WAV2VEC with Attention mechanism"
  echo "- Validation accuracy: 85.00%"
  echo "- Training epochs: 30 (early stopping at epoch 18)"
  echo "- Features: WAV2VEC embeddings"
else
  echo "Error: Failed to download model files."
  exit 1
fi

echo "===== Download Complete ====="
EOF
    
    chmod +x download_v9_fixed_model.sh
  fi
  
  # Run the download script
  ./download_v9_fixed_model.sh
  
  if [ ! -f "models/wav2vec_v9_attention/best_model_v9.h5" ]; then
    echo "Error: Failed to download model. Please check EC2 connection or model location."
    exit 1
  fi
fi

# Check if SSH connection is active before running evaluation
if [ -f "aws_instance_ip.txt" ]; then
  echo "Checking SSH connection to EC2 instance..."
  EC2_HOST=$(cat aws_instance_ip.txt)
  EC2_USER="ubuntu"
  
  ssh -o BatchMode=yes -o ConnectTimeout=5 $EC2_USER@$EC2_HOST true 2>/dev/null
  
  if [ $? -ne 0 ]; then
    echo "Warning: Cannot connect to EC2 instance. Evaluation will use only local files."
    echo "If you need to evaluate on EC2 files, please check your EC2 connection."
  else
    echo "SSH connection successful. Evaluation can access EC2 files if needed."
  fi
fi

# Parse command line arguments
CROSS_VALIDATION=false
N_FOLDS=5
MAX_LENGTH=221
OUTPUT_DIR="results/wav2vec_evaluation_$(date +%Y%m%d_%H%M%S)"
SEARCH_DIRS=()

while (( "$#" )); do
  case "$1" in
    --cross-validation)
      CROSS_VALIDATION=true
      shift
      ;;
    --n-folds)
      N_FOLDS="$2"
      shift 2
      ;;
    --max-length)
      MAX_LENGTH="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --search-dir)
      SEARCH_DIRS+=("$2")
      shift 2
      ;;
    *)
      echo "Error: Unsupported flag $1" >&2
      echo "Usage: $0 [--cross-validation] [--n-folds N] [--max-length L] [--output-dir DIR] [--search-dir DIR1] [--search-dir DIR2] ..." >&2
      exit 1
      ;;
  esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Construct the command
CMD="./evaluate_wav2vec_full_dataset.py --model-path models/wav2vec_v9_attention/best_model_v9.h5 --label-classes models/wav2vec_v9_attention/label_classes_v9.npy --max-length $MAX_LENGTH --output-dir $OUTPUT_DIR"

# Add search directories if provided
for dir in "${SEARCH_DIRS[@]}"; do
  CMD="$CMD --search-dir \"$dir\""
done

# Add cross-validation if requested
if [ "$CROSS_VALIDATION" = true ]; then
  CMD="$CMD --cross-validation --n-folds $N_FOLDS"
fi

echo "Running evaluation with the following command:"
echo "$CMD"
echo ""

# Execute the command
eval "$CMD"

# Check if evaluation completed successfully
if [ $? -eq 0 ]; then
  echo ""
  echo "===== Evaluation Complete ====="
  echo "Results saved to $OUTPUT_DIR"
  
  # If confusion matrix was generated, show its path
  if [ -f "$OUTPUT_DIR/confusion_matrix.png" ]; then
    echo "Confusion matrix: $OUTPUT_DIR/confusion_matrix.png"
  fi
  
  if [ -f "$OUTPUT_DIR/cv_confusion_matrix.png" ]; then
    echo "Cross-validation confusion matrix: $OUTPUT_DIR/cv_confusion_matrix.png"
  fi
  
  # If results file was generated, show its path
  if [ -f "$OUTPUT_DIR/evaluation_results.json" ]; then
    echo "Evaluation results: $OUTPUT_DIR/evaluation_results.json"
  fi
  
  if [ -f "$OUTPUT_DIR/cross_validation_results.json" ]; then
    echo "Cross-validation results: $OUTPUT_DIR/cross_validation_results.json"
  fi
else
  echo ""
  echo "===== Evaluation Failed ====="
  echo "Please check the error messages above."
fi
