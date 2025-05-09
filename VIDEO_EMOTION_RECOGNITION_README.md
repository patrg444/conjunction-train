# Video Emotion Recognition Framework

This project implements a complete video-only emotion recognition model using 3D-ResNet-LSTM architecture trained on both RAVDESS and CREMA-D datasets. The framework extracts spatio-temporal features from video facial expressions and classifies them into six common emotion categories.

## Key Features

- **End-to-end pipeline**: From dataset preparation to model training to evaluation
- **State-of-the-art architecture**: 3D-ResNet-18 backbone + LSTM for temporal modeling
- **Cross-dataset training**: Combined RAVDESS and CREMA-D for better generalization
- **Mixed precision training**: For faster throughput on compatible GPUs
- **Comprehensive monitoring**: Real-time tracking of training progress
- **Robust evaluation**: Confusion matrices, classification reports, and learning curves

## Components

### Scripts

1. **Data Preparation**
   - `scripts/generate_video_manifest.py`: Creates a CSV manifest with train/val/test splits

2. **Model Training**
   - `scripts/train_video_full.py`: Implements the 3D-ResNet-LSTM model and training loop

3. **Utility Scripts**
   - `launch_video_full_training.sh`: Starts training in a tmux session
   - `monitor_video_training.sh`: Monitors training progress in real time
   - `deploy_video_full_training.sh`: Deploys scripts to EC2 instance
   - `download_video_model.sh`: Downloads trained model and results

## Architecture

The model uses a 3D-ResNet-18 backbone to extract spatio-temporal features from video frames, followed by an LSTM to model temporal patterns:

```
Input video (48 frames, 112x112) 
  → 3D ResNet-18 
  → Global Avg Pool 
  → Bidirectional LSTM (128 units) 
  → Dense (128, ReLU) 
  → Dropout (0.5) 
  → Dense (6, Softmax)
```

## Datasets

The framework works with two datasets:

1. **RAVDESS**: 24 actors, ~1,440 video clips, 8 emotions
2. **CREMA-D**: 91 actors, ~7,442 video clips, 6 emotions

The model uses six common emotions: angry, disgust, fearful, happy, neutral, and sad.

## Usage

### Step 1: Deploy scripts to EC2 (Optional)

```bash
# Set up permissions
chmod +x deploy_video_full_training.sh

# Deploy to EC2
./deploy_video_full_training.sh
```

### Step 2: Start training

```bash
# On EC2 instance
cd /home/ubuntu
./launch_video_full_training.sh
```

### Step 3: Monitor training

```bash
# On EC2 instance
cd /home/ubuntu
./monitor_video_training.sh
```

### Step 4: Download results

```bash
# On local machine
./download_video_model.sh
```

## Training Parameters

- **Batch size**: 8
- **Learning rate**: 1e-4 with cosine decay
- **Epochs**: 30
- **Optimizer**: AdamW
- **Loss**: Categorical cross-entropy
- **Data augmentation**: Random horizontal flip, brightness/contrast adjustment

## Performance

The model achieves competitive accuracy on emotion recognition, with performance metrics saved to:

- `metrics.json`: Training/validation loss and accuracy by epoch
- `learning_curves.png`: Visual representation of model convergence
- `confusion_matrix_*.png`: Confusion matrices at different epochs
- `classification_report_*.json`: Precision, recall, and F1-score by class

## Requirements

- Python 3.8+
- PyTorch 1.8+
- CUDA-compatible GPU (recommended)
- Required Python packages: `torch`, `torchvision`, `opencv-python`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`

## Directory Structure

```
/emotion_full_video/                # Main output directory
  └── video_full_[timestamp]/       # Run-specific results
      ├── model_best.pt             # Best checkpoint
      ├── metrics.json              # Training metrics
      ├── learning_curves.png       # Loss/accuracy plots
      ├── confusion_matrix_*.png    # Confusion matrices
      └── classification_report_*.json  # Detailed metrics
```

## Extending the Framework

To adapt this framework to other video classification tasks:

1. Modify `EMOTION_LABELS` in `train_video_full.py`
2. Update manifest generation in `generate_video_manifest.py`
3. Adjust model architecture as needed for your task

## Troubleshooting

- **GPU memory issues**: Reduce batch size or number of frames
- **Training instability**: Lower learning rate or adjust dropout
- **Data loading errors**: Verify dataset paths in config
