# SlowFast Emotion Recognition Model

## Model Overview

The SlowFast emotion recognition model has been successfully trained to 92.9% validation accuracy using only the video modality. This document provides information about the model, its performance, and how to use it.

## Performance

- **Architecture**: SlowFast-R50 (3D CNN backbone)
- **Training Data**: RAVDESS and CREMA-D video emotion datasets
- **Final Validation Accuracy**: 92.9% 
- **Training Duration**: 36 epochs (early-stopped from planned 60 epochs)
- **Classes**: 6 emotion categories (angry, disgust, fearful, happy, neutral, sad)
- **Input**: Video clips of faces expressing emotions

## Model Files

Two versions of the model are available:

1. **Full Checkpoint** (~399 MB)
   - Contains full training state (model weights, optimizer state, scheduler state)
   - Useful for resuming training
   - File: `models/slowfast_emotion_full_checkpoint.pt`

2. **Deployment Model** (~90 MB)
   - Contains only model weights (no optimizer or scheduler state)
   - Optimized for deployment
   - File: `models/slowfast_emotion_video_only_92.9.pt`

## How to Use

### Download and Extract Model

We provide scripts to download the model from EC2 and extract the model weights:

```bash
# Download model from EC2 and extract weights
./download_and_extract_slowfast_model.sh
```

This will create a `models` directory containing both the full checkpoint and the deployment-ready version.

### Load the Model in Python

Here's how to load and use the model in your Python application:

```python
import torch
from scripts.train_slowfast_emotion import EmotionClassifier

# Initialize the model architecture
model = EmotionClassifier(
    num_classes=6,  # 6 emotion categories
    hidden_size=256,
    dropout=0.5,
    use_se=True,     # Use Squeeze-Excitation blocks
    pretrained=False  # No need to load pretrained weights as we'll load our own
)

# Load the trained weights
model.load_state_dict(torch.load('models/slowfast_emotion_video_only_92.9.pt'))
model.eval()  # Set to evaluation mode

# Now you can use the model for inference
# Input shape should be [batch_size, time, channels, height, width]
# Where time is the number of frames (typically 16 or 48)
```

### Inference Example

```python
import torch
import cv2
import numpy as np
from torchvision import transforms

# Define preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Emotion labels
EMOTION_LABELS = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad']

def predict_emotion(model, video_path, num_frames=48, img_size=112):
    """Predict emotion from a video clip."""
    # Load video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames evenly
    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    
    # Extract and preprocess frames
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (img_size, img_size))
            frame_tensor = transform(frame)
            frames.append(frame_tensor)
    cap.release()
    
    # Stack frames and prepare input
    video_tensor = torch.stack(frames).unsqueeze(0)  # Add batch dimension
    
    # Inference
    with torch.no_grad():
        outputs = model(video_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probs, 1)
    
    emotion = EMOTION_LABELS[prediction.item()]
    conf_value = confidence.item()
    
    return emotion, conf_value
```

## Model Architecture Details

The SlowFast model uses:

- SlowFast-R50 backbone with two pathways:
  - A "Slow" pathway with high spatial resolution but low frame rate sampling
  - A "Fast" pathway with lower spatial resolution but higher frame rate sampling
- Squeeze-and-Excitation (SE) blocks in the backbone to enhance feature quality
- Dropout (0.5) for regularization
- AdamW optimizer with weight decay
- One-Cycle LR scheduling
- Label smoothing (0.05)

## Future Improvements

- **Multimodal Fusion**: Adding audio features could further improve accuracy by ~3-4%
- **Model Distillation**: A smaller version could be created for mobile/edge deployment
- **Ensemble Methods**: Using multiple clips during inference could improve robustness
- **TorchScript/ONNX Export**: Converting to these formats would enable deployment on more platforms

## Further Questions

For any issues or questions about the model, please contact the team.
