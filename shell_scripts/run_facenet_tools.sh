#!/bin/bash
# Run Facenet video-only emotion recognition tools
# This script provides easy access to the various debugging, training,
# and evaluation tools for the Facenet video-only LSTM model.

set -e

# Default directories
RAVDESS_DIR="./ravdess_features_facenet"
CREMAD_DIR="./crema_d_features_facenet"
MODEL_DIR="./models/facenet_lstm"
BATCH_SIZE=32

show_help() {
    echo "Facenet Video-Only Emotion Recognition Tools"
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  debug        Run batch inspection tool to analyze features and labels"
    echo "  verify-split Verify training/validation dataset splitting"
    echo "  test-train   Run quick training (3 epochs) to verify pipeline"
    echo "  train        Train the Facenet video-only LSTM model"
    echo "  evaluate     Generate confusion matrix and classification report"
    echo "  deploy-gpu   Deploy and train on GPU-enabled AWS instance"
    echo ""
    echo "Options:"
    echo "  --ravdess-dir DIR  Directory with RAVDESS features (default: $RAVDESS_DIR)"
    echo "  --cremad-dir DIR   Directory with CREMA-D features (default: $CREMAD_DIR)"
    echo "  --model-dir DIR    Directory to save/load model (default: $MODEL_DIR)"
    echo "  --batch-size N     Batch size for training/evaluation (default: $BATCH_SIZE)"
    echo "  --aws-ip IP        AWS instance IP address (for deploy-gpu)"
    echo "  --help             Show this help message and exit"
    echo ""
    echo "Examples:"
    echo "  $0 debug --batch-size 16"
    echo "  $0 train --ravdess-dir /path/to/features --batch-size 64"
    echo "  $0 evaluate --model-dir ./models/best_model"
    echo "  $0 deploy-gpu --aws-ip 1.2.3.4"
}

# Check for --help option first
if [[ "$1" == "--help" ]]; then
    show_help
    exit 0
fi

# Check if no arguments provided
if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

# Parse command
COMMAND=$1
shift

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --ravdess-dir)
            RAVDESS_DIR="$2"
            shift 2
            ;;
        --cremad-dir)
            CREMAD_DIR="$2"
            shift 2
            ;;
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --aws-ip)
            AWS_IP="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Make directories if they don't exist
mkdir -p "$MODEL_DIR"

# Execute the specified command
case $COMMAND in
    debug)
        echo "=== Running Batch Inspection Tool ==="
        echo "RAVDESS directory: $RAVDESS_DIR"
        echo "CREMA-D directory: $CREMAD_DIR"
        echo "Batch size: $BATCH_SIZE"
        echo ""
        python3 scripts/debug_batch_inspection.py \
            --ravdess_dir "$RAVDESS_DIR" \
            --cremad_dir "$CREMAD_DIR" \
            --batch_size "$BATCH_SIZE"
        ;;
        
    verify-split)
        echo "=== Verifying Dataset Split ==="
        echo "RAVDESS directory: $RAVDESS_DIR"
        echo "CREMA-D directory: $CREMAD_DIR"
        echo ""
        python3 scripts/verify_dataset_split.py \
            --ravdess_dir "$RAVDESS_DIR" \
            --cremad_dir "$CREMAD_DIR"
        ;;
    
    test-train)
        echo "=== Quick Test Training (3 Epochs) ==="
        echo "RAVDESS directory: $RAVDESS_DIR"
        echo "CREMA-D directory: $CREMAD_DIR"
        echo "Model directory: $MODEL_DIR"
        echo "Batch size: $BATCH_SIZE"
        echo ""
        python3 scripts/test_train_facenet_lstm.py \
            --ravdess_dir "$RAVDESS_DIR" \
            --cremad_dir "$CREMAD_DIR" \
            --model_dir "$MODEL_DIR" \
            --batch_size "$BATCH_SIZE" \
            --epochs 3
        ;;
    
    train)
        echo "=== Training Facenet Video-Only LSTM Model ==="
        echo "RAVDESS directory: $RAVDESS_DIR"
        echo "CREMA-D directory: $CREMAD_DIR"
        echo "Model directory: $MODEL_DIR"
        echo "Batch size: $BATCH_SIZE"
        echo ""
        python3 scripts/train_video_only_facenet_lstm_fixed.py \
            --ravdess_dir "$RAVDESS_DIR" \
            --cremad_dir "$CREMAD_DIR" \
            --model_dir "$MODEL_DIR" \
            --batch_size "$BATCH_SIZE"
        ;;
    
    evaluate)
        echo "=== Evaluating Facenet Video-Only LSTM Model ==="
        echo "RAVDESS directory: $RAVDESS_DIR"
        echo "CREMA-D directory: $CREMAD_DIR"
        echo "Model path: $MODEL_DIR"
        echo "Batch size: $BATCH_SIZE"
        echo ""
        
        # Check if model exists
        if [ ! -f "$MODEL_DIR/best_model.h5" ]; then
            echo "Error: Model file not found at $MODEL_DIR/best_model.h5"
            exit 1
        fi
        
        python3 scripts/eval_confusion_matrix.py \
            --model_path "$MODEL_DIR/best_model.h5" \
            --ravdess_dir "$RAVDESS_DIR" \
            --cremad_dir "$CREMAD_DIR" \
            --batch_size "$BATCH_SIZE"
        ;;
    
    deploy-gpu)
        echo "=== Deploying to GPU-Enabled AWS Instance ==="
        if [ -z "$AWS_IP" ]; then
            echo "Error: AWS instance IP address required. Use --aws-ip to specify."
            exit 1
        fi
        
        echo "AWS instance IP: $AWS_IP"
        echo ""
        bash aws-setup/deploy_fixed_facenet_gpu.sh --instance "$AWS_IP"
        ;;
    
    *)
        echo "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac

echo ""
echo "Command completed successfully!"
