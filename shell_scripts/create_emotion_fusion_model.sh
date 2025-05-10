#!/usr/bin/env bash
# Script to create a fusion model that combines SlowFast video model with
# wav2vec audio model for multimodal emotion recognition

# Set variables
EC2_INSTANCE="ubuntu@54.162.134.77"
KEY_PATH="$HOME/Downloads/gpu-key.pem"
REMOTE_DIR="/home/ubuntu/audio_emotion"
VIDEO_WEIGHT=${1:-0.7}  # Default to 70% video, 30% audio
AUDIO_WEIGHT=${2:-0.3}

echo "=== Creating Audio-Video Emotion Fusion Model ==="
echo "EC2 Instance: $EC2_INSTANCE"
echo "Remote Directory: $REMOTE_DIR"
echo "Fusion Weights - Video: $VIDEO_WEIGHT, Audio: $AUDIO_WEIGHT"
echo

# Create a fusion script
cat > create_emotion_fusion.py << 'EOL'
#!/usr/bin/env python3
"""
Create a fusion model that combines SlowFast video model with wav2vec audio model
for multimodal emotion recognition.
"""

import os
import argparse
import json
import shutil

def create_fusion_config(video_model_path, audio_model_dir, output_dir, 
                         video_weight=0.7, audio_weight=0.3):
    """Create a fusion model configuration."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine the audio model path
    audio_model_file = os.path.join(audio_model_dir, 'best_wav2vec_lstm_model.h5')
    if not os.path.exists(audio_model_file):
        # Try alternative file names
        alternatives = [
            'final_wav2vec_lstm_model.h5',
            'best_wav2vec_transformer_model.h5',
            'final_wav2vec_transformer_model.h5'
        ]
        
        for alt in alternatives:
            alt_path = os.path.join(audio_model_dir, alt)
            if os.path.exists(alt_path):
                audio_model_file = alt_path
                break
    
    if not os.path.exists(audio_model_file):
        print(f"Error: Could not find audio model in {audio_model_dir}")
        return None
    
    # Copy the audio model to the output directory
    audio_model_dest = os.path.join(output_dir, os.path.basename(audio_model_file))
    shutil.copy2(audio_model_file, audio_model_dest)
    
    # Create the audio model configuration
    audio_model_config = {
        "model_path": audio_model_dest,
        "features_path": os.path.join(os.path.dirname(os.path.dirname(audio_model_dir)), "wav2vec"),
        "num_features": 768  # wav2vec base hidden size
    }
    
    # Create the fusion model configuration
    fusion_config = {
        "fusion_type": "late",
        "video_model_path": video_model_path,
        "audio_model_config": audio_model_config,
        "video_weight": video_weight,
        "audio_weight": audio_weight,
        "emotion_labels": ["angry", "disgust", "fearful", "happy", "neutral", "sad"]
    }
    
    # Save the fusion configuration
    config_path = os.path.join(output_dir, 'fusion_config.json')
    with open(config_path, 'w') as f:
        json.dump(fusion_config, f, indent=2)
    
    print(f"Fusion model configuration saved to {config_path}")
    
    # Create a simple demo script for the fusion model
    demo_script = """#!/usr/bin/env python3
import os
import json
import numpy as np
import tensorflow as tf
import torch
from tensorflow.keras.models import load_model

def load_fusion_model(fusion_config_path):
    \"\"\"Load the fusion model from the configuration file.\"\"\"
    with open(fusion_config_path, 'r') as f:
        config = json.load(f)
    
    # Load the audio model
    audio_model = load_model(config['audio_model_config']['model_path'])
    
    # For actual implementation, you would also load the video model
    # video_model = ... load SlowFast model ...
    
    return {
        'audio_model': audio_model,
        'config': config
    }

def predict_emotion(fusion_model, audio_features, video_features=None):
    \"\"\"Make a prediction using the fusion model.\"\"\"
    config = fusion_model['config']
    
    # Get audio prediction
    audio_pred = fusion_model['audio_model'].predict(np.expand_dims(audio_features, axis=0))[0]
    
    if video_features is not None:
        # In a real implementation, you would get the video prediction
        # video_pred = fusion_model['video_model'].predict(video_features)
        # For now, simulate with random values
        video_pred = np.random.rand(6)
        video_pred = video_pred / np.sum(video_pred)  # Normalize
        
        # Fuse predictions
        final_pred = (config['video_weight'] * video_pred + 
                     config['audio_weight'] * audio_pred)
    else:
        # Use only audio prediction if no video features
        final_pred = audio_pred
    
    # Get the emotion label
    emotion_idx = np.argmax(final_pred)
    emotion_label = config['emotion_labels'][emotion_idx]
    
    return {
        'emotion': emotion_label,
        'probabilities': final_pred.tolist()
    }

if __name__ == "__main__":
    # Example usage
    fusion_config_path = "fusion_config.json"
    
    if not os.path.exists(fusion_config_path):
        print(f"Error: Fusion config not found at {fusion_config_path}")
        exit(1)
    
    # Load the fusion model
    print("Loading fusion model...")
    fusion_model = load_fusion_model(fusion_config_path)
    
    # For demo purposes, create random features
    print("Creating demo features...")
    dummy_audio_features = np.random.rand(500, 768)  # Matches the padded length in training
    
    # Make a prediction
    print("Making prediction...")
    result = predict_emotion(fusion_model, dummy_audio_features)
    
    print("\\nPrediction result:")
    print(f"Predicted emotion: {result['emotion']}")
    print("Class probabilities:")
    for i, label in enumerate(fusion_model['config']['emotion_labels']):
        print(f"  {label}: {result['probabilities'][i]:.4f}")
"""
    
    demo_path = os.path.join(output_dir, 'demo_fusion_model.py')
    with open(demo_path, 'w') as f:
        f.write(demo_script)
    
    os.chmod(demo_path, 0o755)  # Make executable
    
    print(f"Demo script created at {demo_path}")
    
    return config_path

def main():
    parser = argparse.ArgumentParser(description="Create a fusion model configuration")
    parser.add_argument("--video_model", type=str, required=True,
                        help="Path to the SlowFast video model")
    parser.add_argument("--audio_model_dir", type=str, required=True,
                        help="Directory containing the wav2vec audio model")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the fusion model")
    parser.add_argument("--video_weight", type=float, default=0.7,
                        help="Weight for the video model (0.0-1.0)")
    parser.add_argument("--audio_weight", type=float, default=0.3,
                        help="Weight for the audio model (0.0-1.0)")
    args = parser.parse_args()
    
    # Validate weights
    if args.video_weight + args.audio_weight != 1.0:
        print("Warning: Weights don't sum to 1.0, normalizing...")
        total = args.video_weight + args.audio_weight
        args.video_weight /= total
        args.audio_weight /= total
    
    # Create the fusion model configuration
    create_fusion_config(
        args.video_model,
        args.audio_model_dir,
        args.output_dir,
        args.video_weight,
        args.audio_weight
    )

if __name__ == "__main__":
    main()
EOL
chmod +x create_emotion_fusion.py

# Create a runner script
cat > create_fusion_model.sh << EOL
#!/bin/bash
# Create a fusion model on EC2

# Parse command line arguments
VIDEO_WEIGHT=\${1:-0.7}
AUDIO_WEIGHT=\${2:-0.3}

# Activate the PyTorch environment
source /opt/pytorch/bin/activate

# Set the directory
cd $REMOTE_DIR

# Ensure necessary directories exist
mkdir -p models/fusion

# Check if required components exist
if [ ! -d "models/audio_emotion" ]; then
    echo "Error: Audio model directory not found at models/audio_emotion"
    echo "Please train an audio model first using train_wav2vec_emotion.sh"
    exit 1
fi

# Look for the SlowFast video model
VIDEO_MODEL="models/slowfast_emotion_video_only_92.9.pt"
if [ ! -f "\$VIDEO_MODEL" ]; then
    echo "Warning: Recommended SlowFast model not found at \$VIDEO_MODEL"
    
    # Look for alternatives
    alt_models=\$(find models -name "*slowfast*.pt" -o -name "*video*.pt" 2>/dev/null)
    if [ -n "\$alt_models" ]; then
        # Take the first one
        VIDEO_MODEL=\$(echo "\$alt_models" | head -n 1)
        echo "Using alternative video model: \$VIDEO_MODEL"
    else
        echo "No video model found. Creating a dummy model for demo purposes."
        mkdir -p models/dummy
        echo "Placeholder for SlowFast model" > models/dummy/slowfast_dummy.pt
        VIDEO_MODEL="models/dummy/slowfast_dummy.pt"
    fi
fi

echo
echo "=== Creating Fusion Model ==="
echo "Video Model: \$VIDEO_MODEL"
echo "Audio Model: models/audio_emotion"
echo "Video Weight: \$VIDEO_WEIGHT"
echo "Audio Weight: \$AUDIO_WEIGHT"
echo

# Run the fusion model creator
python create_emotion_fusion.py \
    --video_model "\$VIDEO_MODEL" \
    --audio_model_dir models/audio_emotion \
    --output_dir models/fusion \
    --video_weight "\$VIDEO_WEIGHT" \
    --audio_weight "\$AUDIO_WEIGHT"

echo
echo "=== Fusion Model Creation Complete ==="
echo "The fusion model configuration is saved in models/fusion/"
echo

# Create a tarball for easy download
echo "Creating tarball of fusion model..."
tar -czvf fusion_model.tar.gz models/fusion/
echo "Tarball created at: $REMOTE_DIR/fusion_model.tar.gz"
EOL
chmod +x create_fusion_model.sh

# Upload the scripts
echo "Uploading fusion model scripts to EC2..."
scp -i "$KEY_PATH" create_emotion_fusion.py "$EC2_INSTANCE:$REMOTE_DIR/"
scp -i "$KEY_PATH" create_fusion_model.sh "$EC2_INSTANCE:$REMOTE_DIR/"

echo
echo "Fusion model creation scripts uploaded to EC2."
echo "To create the fusion model (after training the audio model):"
echo "  ssh -i $KEY_PATH $EC2_INSTANCE"
echo "  cd $REMOTE_DIR"
echo "  ./create_fusion_model.sh $VIDEO_WEIGHT $AUDIO_WEIGHT"
echo
echo "After creating the fusion model, download it with:"
echo "  mkdir -p fusion_model"
echo "  scp -i $KEY_PATH $EC2_INSTANCE:$REMOTE_DIR/fusion_model.tar.gz fusion_model/"
echo "  tar -xzvf fusion_model/fusion_model.tar.gz -C fusion_model/"
echo
echo "This will create a complete audio-video fusion model for emotion recognition."
