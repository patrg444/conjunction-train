# Improved Humor Detection with DeBERTa

This document describes the improved humor detection system implemented using Microsoft's DeBERTa model, along with tools for training and deployment.

## Overview

The system provides end-to-end capabilities for training, evaluating, and deploying humor detection models:

1. **Training**: Train DeBERTa models from scratch on humor datasets with hyperparameter tuning
2. **Deployment**: Tools to easily deploy trained models to AWS EC2 instances
3. **Inference**: Use the trained models for humor classification on new text

## Training Scripts

### Training DeBERTa Models

The enhanced training script supports both hyperparameter grid search and single model training with extensive configuration options:

```bash
# Run a grid search across different learning rates
./shell/train_deberta_humor.sh grid

# Train a single model with specified learning rate
./shell/train_deberta_humor.sh single 2e-5

# Dry-run mode to preview commands without executing them
./shell/train_deberta_humor.sh grid --dry-run
```

### Command-line Options

The training script supports these options:

- `--model NAME`: Model name/path (default: microsoft/deberta-v3-base)
- `--batch_size N`: Batch size (default: 32)
- `--epochs N`: Number of training epochs (default: 3)
- `--weight_decay N`: Weight decay value (default: 0.01)
- `--dropout N`: Dropout rate (default: 0.2)
- `--max_length N`: Maximum sequence length (default: a128)
- `--gpus N`: Number of GPUs to use (default: 1)
- `--no_fp16`: Disable mixed precision training
- `--workers N`: Number of dataloader workers (default: 4)
- `--scheduler TYPE`: Learning rate scheduler [cosine|linear] (default: cosine)
- `--grad_clip N`: Gradient clipping value (default: 1.0)
- `--class_balanced`: Use class-balanced loss
- `--base_dir DIR`: Base directory for logs (default: training_logs_humor)
- `--dry-run`: Preview the commands without executing them

### Examples

```bash
# Run grid search with smaller model and more epochs
./shell/train_deberta_humor.sh grid --model microsoft/deberta-v3-small --epochs 5

# Train with specific options
./shell/train_deberta_humor.sh single 2e-5 --batch_size 16 --max_length 256

# Preview commands for a grid search
./shell/train_deberta_humor.sh grid --dry-run
```

## Deployment Script

The `upload_deberta_files.sh` script helps deploy trained models to an EC2 instance:

```bash
# Upload with default options
./upload_deberta_files.sh

# Show commands without executing (dry run)
./upload_deberta_files.sh --dry-run

# Upload model with manifests
./upload_deberta_files.sh --include_manifests
```

### Deployment Options

- `--help`, `-h`: Show help message and exit
- `--dry-run`: Print commands without executing them
- `--key PATH`: Path to SSH key (default: /Users/patrickgloria/Downloads/gpu-key.pem)
- `--host HOST`: EC2 instance hostname or IP (using aws_instance_ip.txt by default)
- `--model_dir DIR`: Local directory containing trained model (default: training_logs_humor)
- `--remote_dir DIR`: Remote directory on EC2 instance (default: ~/humor_models)
- `--include_manifests`: Also upload humor manifests and datasets
- `--include_source`: Also upload source code files

## Dataset Format

The humor dataset should be in CSV format with at least the following columns:
- Text content (jokes, punchlines, etc.)
- Labels (0 for non-humorous, 1 for humorous)

Example manifest file used by default is `datasets/manifests/humor/ur_funny_train_humor_cleaned.csv`.

## Model Architecture

The implementation uses DeBERTa-v3, which provides superior performance compared to BERT and RoBERTa on many language tasks, particularly for nuanced classification tasks like humor detection.

### Key Features

- Pre-trained on a large text corpus with disentangled attention mechanism
- Fine-tuned specifically for humor classification
- Supports mixed precision training for faster execution
- Implements effective regularization through weight decay and dropout

## Performance

Using the improved training and deployment workflow with DeBERTa-v3-base, we typically see:

- Accuracy: 85-92% (depending on dataset)
- F1 Score: 0.83-0.90
- Training time: ~1-2 hours on an AWS g4dn.xlarge instance

## Next Steps

Possible improvements for future work:

1. Ensemble models combining multiple pretrained transformers
2. Integrating contextual features beyond just text
3. Multi-task learning to capture related aspects of humor
4. Distillation for faster inference in production

## Troubleshooting

If you encounter issues:

1. Check that the manifest files exist and are correctly formatted
2. Verify AWS credentials and EC2 connectivity if using deployment
3. Try dry-run mode to preview and validate commands before execution
4. Check GPU availability and CUDA compatibility if using GPU training
