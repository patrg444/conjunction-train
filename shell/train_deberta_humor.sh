#!/bin/bash
set -e

# Function to show help message
show_help() {
  cat << EOF
Usage: $(basename "$0") [MODE] [OPTIONS]

MODES:
  grid                Run a grid search with multiple learning rates (5e-6, 1e-5, 2e-5)
  single <lr>         Run a single model with specified learning rate

OPTIONS:
  --help, -h          Show this help message and exit
  --dry-run           Print commands that would be executed without running them
  --model NAME        Model name/path (default: microsoft/deberta-v3-base)
  --batch_size N      Batch size (default: 32)
  --epochs N          Number of training epochs (default: 3)
  --weight_decay N    Weight decay value (default: 0.01)
  --dropout N         Dropout rate (default: 0.2)
  --max_length N      Maximum sequence length (default: 128)
  --gpus N            Number of GPUs to use (default: 1)
  --no_fp16           Disable mixed precision training
  --workers N         Number of dataloader workers (default: 4)
  --scheduler TYPE    Learning rate scheduler [cosine|linear] (default: cosine)
  --grad_clip N       Gradient clipping value (default: 1.0)
  --class_balanced    Use class-balanced loss
  --base_dir DIR      Base directory for logs (default: training_logs_humor)

Examples:
  $(basename "$0") grid                    # Run grid search with default parameters
  $(basename "$0") grid --model microsoft/deberta-v3-small --epochs 5
  $(basename "$0") single 2e-5             # Train with LR=2e-5
  $(basename "$0") grid --dry-run          # Show commands without running
EOF
  exit 0
}

# Dry-run mode flag (0=normal execution, 1=print commands only)
DRY_RUN=0

# Default values for options
MODEL_NAME="microsoft/deberta-v3-base"
BATCH_SIZE=32
NUM_WORKERS=4
EPOCHS=3
WEIGHT_DECAY=0.01
DROPOUT=0.2
MAX_LENGTH=128
FP16="--fp16"
GPUS=1
SCHEDULER="cosine"
GRAD_CLIP=1.0
CLASS_BALANCED=""
BASE_DIR="training_logs_humor"

# First, process any options before the mode
MODE=""
LEARNING_RATE=""

# First check for --help or mode
if [[ $# -gt 0 ]]; then
  case $1 in
    -h|--help)
      show_help
      ;;
    grid)
      MODE="grid"
      shift
      ;;
    single)
      MODE="single"
      if [[ $# -gt 1 ]]; then
        LEARNING_RATE="$2"
        shift 2
      else
        shift
      fi
      ;;
  esac
fi

# Process remaining arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      show_help
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --model)
      MODEL_NAME="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --weight_decay)
      WEIGHT_DECAY="$2"
      shift 2
      ;;
    --dropout)
      DROPOUT="$2"
      shift 2
      ;;
    --max_length)
      MAX_LENGTH="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --no_fp16)
      FP16=""
      shift
      ;;
    --scheduler)
      SCHEDULER="$2"
      shift 2
      ;;
    --grad_clip)
      GRAD_CLIP="$2"
      shift 2
      ;;
    --class_balanced)
      CLASS_BALANCED="--class_balanced_loss"
      shift
      ;;
    --base_dir)
      BASE_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Run a grid search over learning rates
run_lr_grid_search() {
  echo "====================================================="
  echo "🧠 Starting learning rate grid search for humor detection"
  echo "====================================================="
  echo "Model: $MODEL_NAME"
  echo "Batch size: $BATCH_SIZE"
  echo "Epochs: $EPOCHS"
  echo "Weight decay: $WEIGHT_DECAY"
  echo "Dropout: $DROPOUT"
  echo "Scheduler: $SCHEDULER"
  echo "Gradient clipping: $GRAD_CLIP"
  echo "Class balanced loss: $CLASS_BALANCED"
  echo "====================================================="

  # Grid of learning rates to try
  LR_VALUES=("5e-6" "1e-5" "2e-5")
  
  for lr in "${LR_VALUES[@]}"; do
    echo "Training with learning rate: $lr"
    exp_name="${MODEL_NAME//\//_}_lr_${lr}"
    
    cmd="python train_deberta_humor.py \
      --train_manifest datasets/manifests/humor/ur_funny_train_humor_cleaned.csv \
      --val_manifest datasets/manifests/humor/ur_funny_val_humor_cleaned.csv \
      --model_name \"$MODEL_NAME\" \
      --max_length $MAX_LENGTH \
      --batch_size $BATCH_SIZE \
      --learning_rate $lr \
      --epochs $EPOCHS \
      --num_workers $NUM_WORKERS \
      --weight_decay $WEIGHT_DECAY \
      --dropout $DROPOUT \
      --scheduler $SCHEDULER \
      --grad_clip $GRAD_CLIP \
      $CLASS_BALANCED \
      --log_dir $BASE_DIR \
      --exp_name \"$exp_name\" \
      --gpus $GPUS $FP16"
    
    if [[ $DRY_RUN -eq 1 ]]; then
      echo "DRY RUN: Command that would be executed:"
      echo "$cmd"
    else
      eval "$cmd"
      echo "Completed training with learning rate: $lr"
      echo "---------------------------------------------------"
    fi
  done

  if [[ $DRY_RUN -eq 0 ]]; then
    echo "Learning rate grid search completed."
  else
    echo "DRY RUN: Learning rate grid search commands displayed."
  fi
  echo "====================================================="
}

# Run a single model with specific learning rate
run_single_model() {
  local lr="1e-5"  # Default learning rate
  if [[ $# -gt 0 ]]; then
    lr="$1"
  fi
  
  echo "====================================================="
  echo "🧠 Training humor detection model with fixed learning rate"
  echo "====================================================="
  echo "Model: $MODEL_NAME"
  echo "Learning rate: $lr"
  echo "Batch size: $BATCH_SIZE"
  echo "====================================================="
  
  exp_name="${MODEL_NAME//\//_}_single"
  
  cmd="python train_deberta_humor.py \
    --train_manifest datasets/manifests/humor/ur_funny_train_humor_cleaned.csv \
    --val_manifest datasets/manifests/humor/ur_funny_val_humor_cleaned.csv \
    --model_name \"$MODEL_NAME\" \
    --max_length $MAX_LENGTH \
    --batch_size $BATCH_SIZE \
    --learning_rate $lr \
    --epochs $EPOCHS \
    --num_workers $NUM_WORKERS \
    --weight_decay $WEIGHT_DECAY \
    --dropout $DROPOUT \
    --scheduler $SCHEDULER \
    --grad_clip $GRAD_CLIP \
    $CLASS_BALANCED \
    --log_dir $BASE_DIR \
    --exp_name \"$exp_name\" \
    --gpus $GPUS $FP16"
  
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "DRY RUN: Command that would be executed:"
    echo "$cmd"
  else
    eval "$cmd"
  fi
}

# Execute the selected mode
if [[ "$MODE" == "grid" ]]; then
  run_lr_grid_search
elif [[ "$MODE" == "single" ]]; then
  run_single_model "$LEARNING_RATE"
elif [[ -z "$MODE" ]]; then
  # Default: run grid search if no mode specified
  run_lr_grid_search
fi

if [[ $DRY_RUN -eq 1 ]]; then
  echo "DRY RUN: Command preview completed. No models were trained."
else
  echo "Text-Humor training completed. Results are in: $BASE_DIR"
fi
