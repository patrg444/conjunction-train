#!/usr/bin/env bash
# Create a fusion model workflow combining SlowFast video and wav2vec audio features

# Set up directories
mkdir -p models/wav2vec
mkdir -p models/fusion

# Step 1: Download the SlowFast model (using previously created script)
echo "Step 1: Downloading SlowFast model..."
./download_and_extract_slowfast_model.sh

# Step 2: Create a script to extract wav2vec embeddings
cat > extract_wav2vec_features.py << 'EOPY'
#!/usr/bin/env python3
"""
Extract wav2vec 2.0 features from RAVDESS and CREMA-D audio for emotion recognition.
"""
import os
import sys
import glob
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm

# Map emotions to consistent integer labels
EMOTION_MAP = {
    'angry': 0,
    'disgust': 1, 
    'fearful': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5
}

def extract_features(audio_path, processor, model, device):
    """Extract wav2vec features from audio file."""
    try:
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:  # Convert to mono if stereo
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Process with wav2vec
        inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Extract features
        with torch.no_grad():
            outputs = model(**inputs)
            features = outputs.last_hidden_state.cpu().numpy().squeeze()
            
        return features
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading wav2vec model...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    model.eval()
    
    # Create output directory
    os.makedirs("models/wav2vec", exist_ok=True)
    
    # Process a few sample files from each dataset for testing
    ravdess_files = glob.glob("downsampled_videos/RAVDESS/*/*.wav")[:5]
    cremad_files = glob.glob("downsampled_videos/CREMA-D-audio-complete/*.wav")[:5]
    
    for file_path in tqdm(ravdess_files + cremad_files):
        try:
            # Get emotion from filename
            filename = os.path.basename(file_path)
            
            if '-' in filename:  # RAVDESS format
                parts = filename.split('-')
                emotion_code = parts[2]
                emotion_map = {'01': 'neutral', '02': 'neutral', '03': 'happy', 
                               '04': 'sad', '05': 'angry', '06': 'fearful', '07': 'disgust'}
                emotion = emotion_map.get(emotion_code, 'unknown')
            else:  # CREMA-D format
                parts = filename.split('_')
                emotion = parts[2].lower()
            
            if emotion not in EMOTION_MAP:
                continue
                
            # Extract features
            features = extract_features(file_path, processor, model, device)
            if features is None:
                continue
                
            # Save features
            output_file = os.path.join("models/wav2vec", f"{os.path.splitext(filename)[0]}.npz")
            np.savez(output_file, 
                     wav2vec_features=features, 
                     emotion=emotion,
                     label=EMOTION_MAP[emotion])
            
            print(f"Processed {filename} ({emotion})")
        except Exception as e:
            print(f"Error with {file_path}: {e}")

if __name__ == "__main__":
    main()
EOPY
chmod +x extract_wav2vec_features.py

# Step 3: Create a fusion model script
cat > create_emotion_fusion.py << 'EOPY'
#!/usr/bin/env python3
"""
Create a fusion model that combines SlowFast video and wav2vec audio models.
"""
import os
import numpy as np
import torch
import tensorflow as tf
from tensorflow import keras
import argparse
import json

# Configure GPU memory growth for TensorFlow
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Emotion labels - must be consistent with both models
EMOTION_LABELS = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad']

class LateFusionModel:
    """Late fusion (ensemble) of video and audio models."""
    
    def __init__(self, video_model_path, audio_model_path, video_weight=0.7, audio_weight=0.3):
        self.video_weight = video_weight
        self.audio_weight = audio_weight
        
        # Normalize weights
        total = video_weight + audio_weight
        self.video_weight /= total
        self.audio_weight /= total
        
        # Load models (placeholders for actual implementation)
        print(f"Loading video model from {video_model_path}")
        # self.video_model = load_video_model(video_model_path)
        
        print(f"Loading audio model from {audio_model_path}")
        # self.audio_model = load_audio_model(audio_model_path)
    
    def predict(self, video_input, audio_input):
        """Make predictions using both models and combine results."""
        # Placeholder implementation
        video_probs = np.array([0.1, 0.05, 0.05, 0.3, 0.4, 0.1])  # Example output
        audio_probs = np.array([0.15, 0.05, 0.1, 0.2, 0.3, 0.2])  # Example output
        
        # Weighted average
        combined_probs = self.video_weight * video_probs + self.audio_weight * audio_probs
        
        # Get prediction
        pred_idx = np.argmax(combined_probs)
        pred_label = EMOTION_LABELS[pred_idx]
        confidence = combined_probs[pred_idx]
        
        return pred_label, confidence, {label: float(prob) for label, prob in zip(EMOTION_LABELS, combined_probs)}

def save_fusion_config(config, output_path):
    """Save fusion configuration to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Fusion configuration saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Create emotion recognition fusion model")
    parser.add_argument("--video_model", type=str, default="models/slowfast_emotion_video_only_92.9.pt",
                        help="Path to SlowFast video model")
    parser.add_argument("--audio_model", type=str, default="models/wav2vec/best_audio_model.h5",
                        help="Path to wav2vec audio model")
    parser.add_argument("--video_weight", type=float, default=0.7,
                        help="Weight for video model (0.0-1.0)")
    parser.add_argument("--audio_weight", type=float, default=0.3,
                        help="Weight for audio model (0.0-1.0)")
    parser.add_argument("--output_dir", type=str, default="models/fusion",
                        help="Output directory for fusion model")
    args = parser.parse_args()
    
    # Create fusion model (mostly a configuration at this point)
    fusion_model = LateFusionModel(
        video_model_path=args.video_model,
        audio_model_path=args.audio_model,
        video_weight=args.video_weight,
        audio_weight=args.audio_weight,
    )
    
    # Save configuration
    config = {
        "fusion_type": "late",
        "video_model_path": args.video_model,
        "audio_model_path": args.audio_model,
        "video_weight": args.video_weight,
        "audio_weight": args.audio_weight,
        "emotion_labels": EMOTION_LABELS
    }
    save_fusion_config(config, os.path.join(args.output_dir, "fusion_config.json"))
    
    print("Fusion model created")
    print(f"- Video model: {args.video_model} (weight: {args.video_weight:.2f})")
    print(f"- Audio model: {args.audio_model} (weight: {args.audio_weight:.2f})")
    print("To use this model:")
    print("1. Load both video and audio models")
    print("2. Process inputs with respective models")
    print("3. Combine predictions with weighted average")

if __name__ == "__main__":
    main()
EOPY
chmod +x create_emotion_fusion.py

# Step 4: Create a fusion model demo script
cat > demo_fusion_model.py << 'EOPY'
#!/usr/bin/env python3
"""
Demo script that shows how to use the fusion model for emotion recognition.
"""
import os
import sys
import json
import numpy as np
import torch
import tensorflow as tf
import argparse
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import cv2

# Load fusion configuration
def load_fusion_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Demo fusion model for emotion recognition")
    parser.add_argument("--config", type=str, default="models/fusion/fusion_config.json",
                        help="Path to fusion configuration")
    parser.add_argument("--video", type=str, required=True,
                        help="Path to video file for analysis")
    parser.add_argument("--audio", type=str, default=None,
                        help="Path to audio file for analysis (optional, extracted from video if not provided)")
    args = parser.parse_args()
    
    # Load configuration
    config = load_fusion_config(args.config)
    print(f"Loaded fusion configuration: {config['fusion_type']} fusion")
    
    # Placeholder for model loading and inference
    print(f"Video model: {config['video_model_path']}")
    print(f"Audio model: {config['audio_model_path']}")
    print(f"Weights: Video={config['video_weight']}, Audio={config['audio_weight']}")
    
    # Simulate prediction
    emotion_labels = config['emotion_labels']
    video_probs = np.array([0.1, 0.05, 0.05, 0.3, 0.4, 0.1])  # Example
    audio_probs = np.array([0.15, 0.05, 0.1, 0.2, 0.3, 0.2])  # Example
    
    # Weighted average
    combined_probs = config['video_weight'] * video_probs + config['audio_weight'] * audio_probs
    
    # Get prediction
    pred_idx = np.argmax(combined_probs)
    pred_label = emotion_labels[pred_idx]
    confidence = combined_probs[pred_idx]
    
    print(f"\nPredicted emotion: {pred_label} (confidence: {confidence:.2f})")
    print("\nProbabilities for each emotion:")
    for label, prob in zip(emotion_labels, combined_probs):
        print(f"- {label}: {prob:.4f}")

if __name__ == "__main__":
    main()
EOPY
chmod +x demo_fusion_model.py

# Step 5: Create a fusion workflow script
cat > run_fusion_workflow.sh << 'EOF'
#!/usr/bin/env bash
# Run the complete emotion fusion workflow

# Step 1: Make sure SlowFast model is downloaded
if [ ! -f "models/slowfast_emotion_video_only_92.9.pt" ]; then
    echo "Downloading SlowFast model..."
    ./download_and_extract_slowfast_model.sh
fi

# Step 2: Extract wav2vec features for a sample
echo "Extracting wav2vec features..."
python extract_wav2vec_features.py

# Step 3: Create fusion model configuration
echo "Creating fusion model..."
python create_emotion_fusion.py --video_weight 0.7 --audio_weight 0.3

# Step 4: Run a demo on a test video
echo "Running fusion model demo..."
# Find a test video
TEST_VIDEO=$(find downsampled_videos -name "*.mp4" | head -n 1)
if [ -n "$TEST_VIDEO" ]; then
    python demo_fusion_model.py --video "$TEST_VIDEO"
else
    echo "No test video found. Please specify a video path manually."
fi

echo "Fusion workflow complete!"
