# Enhanced Multimodal Emotion Recognition System

This project implements a deep learning-based multimodal emotion recognition system with enhanced models designed to combat overfitting and improve generalization performance.

## Key Improvements

We've made several significant improvements to the emotion recognition system:

1. **Increased Context Window Size**:
   - Extended window size from 1.0s to 3.5s for better emotional context
   - Increased minimum segments target from 250 to 1000 for more training data

2. **Enhanced Models with Anti-Overfitting Techniques**:
   - Added L2 regularization to all layers (kernel, recurrent, and bias)
   - Increased dropout rates (from 0.3 to 0.4-0.5)
   - Added gradient clipping to prevent exploding gradients
   - Applied feature normalization using z-score standardization
   - Modified learning rate scheduling for more gradual learning rate decay
   - Extended early stopping patience
   - Reduced batch size to introduce beneficial noise

3. **Simplified Pipeline**:
   - Created an all-in-one pipeline script to execute the entire workflow
   - Added extensive command-line options for customization
   - Improved error handling and logging
   - Added visualization of features and model performance

## Models

Two enhanced neural network architectures are available:

1. **Enhanced Branched Model**:
   - Multiple LSTM branches for both modalities
   - Cross-modal attention mechanisms
   - Hierarchical fusion of modalities
   - Additional convolutional branch for audio

2. **Enhanced Dual-Stream Model**:
   - Simpler architecture with parallel LSTMs
   - Strong regularization for generalization
   - Early fusion of modalities

## Quick Start

To run the complete pipeline with default settings:

```bash
python scripts/run_enhanced_pipeline.py
```

This will:
1. Process the RAVDESS dataset with 3.5s windows
2. Train both enhanced models
3. Evaluate performance and save results

## Customizing the Pipeline

The pipeline offers several command-line options:

```
usage: run_enhanced_pipeline.py [-h] [--dataset DATASET] [--features-dir FEATURES_DIR] 
                             [--min-segments MIN_SEGMENTS] [--window-size WINDOW_SIZE] 
                             [--workers WORKERS] [--skip-processing]
                             [--models {both,branched,dual_stream}] [--epochs EPOCHS] 
                             [--batch-size BATCH_SIZE] [--no-attention]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Directory containing RAVDESS dataset
  --features-dir FEATURES_DIR
                        Directory to save processed features
  --min-segments MIN_SEGMENTS
                        Minimum number of segments to generate
  --window-size WINDOW_SIZE
                        Time window size in seconds (default: 3.5 seconds)
  --workers WORKERS     Number of parallel workers for processing
  --skip-processing     Skip dataset processing step (use existing features)
  --models {both,branched,dual_stream}
                        Which model(s) to train
  --epochs EPOCHS       Number of training epochs
  --batch-size BATCH_SIZE
                        Training batch size
  --no-attention        Disable attention mechanisms in branched model
```

### Examples

Train only the branched model using existing processed features:
```bash
python scripts/run_enhanced_pipeline.py --skip-processing --models branched
```

Process dataset with custom window size and train dual-stream model:
```bash
python scripts/run_enhanced_pipeline.py --window-size 4.0 --models dual_stream
```

## Individual Components

If you prefer to run components separately:

1. **Process dataset**:
   ```bash
   python scripts/process_ravdess_dataset.py --output processed_features_3_5s --window-size 3.5
   ```

2. **Train enhanced branched model**:
   ```bash
   python scripts/train_branched_enhanced.py --data_dir processed_features_3_5s
   ```

3. **Train enhanced dual-stream model**:
   ```bash
   python scripts/train_dual_stream_enhanced.py --data_dir processed_features_3_5s
   ```

## Expected Output

After running the pipeline, you should see:

- Processed features in the specified directory
- Model files in `models/branched_enhanced/` and `models/dual_stream_enhanced/`
- Evaluation results in `model_evaluation/branched_enhanced/` and `model_evaluation/dual_stream_enhanced/`
- Detailed logs of the training and evaluation process

## Results

The enhanced models should significantly reduce overfitting compared to the baseline models, resulting in:

- Better generalization to unseen data
- More robust performance across different emotions
- Higher F1-scores, especially for under-represented emotions
- Smoother convergence during training
