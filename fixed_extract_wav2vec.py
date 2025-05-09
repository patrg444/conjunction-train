#!/usr/bin/env python3
"""
Extract wav2vec features directly from MP4/FLV files in the RAVDESS and CREMA-D datasets.
This version works directly with video files instead of requiring audio extraction.
"""

import os
import sys
import glob
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm
import tensorflow as tf
import subprocess
import tempfile
import json
import argparse
import random

# Configure GPU memory growth for TensorFlow
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Emotion mapping
RAVDESS_EMOTION_MAP = {
    '01': 'neutral',
    '02': 'calm',     # Also mapped to neutral for 6-class setup
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised' # Not used in our 6-class setup
}

EMOTION_CLASS_MAP = {
    'neutral': 4,
    'calm': 4,      # Map to neutral
    'happy': 3,
    'sad': 5,
    'angry': 0,
    'fearful': 2,
    'disgust': 1,
    'surprised': None  # Not used
}

def extract_audio_with_ffmpeg(video_path, output_path=None):
    """Extract audio from video file using ffmpeg."""
    try:
        # If output_path is not specified, use temp directory
        if output_path is None:
            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            output_path = temp_file.name
            temp_file.close()
        
        # Use ffmpeg to extract audio
        cmd = ['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', '-f', 'wav', output_path, '-y', '-loglevel', 'error']
        subprocess.run(cmd, check=True)
        
        return output_path
    except Exception as e:
        print(f"Error extracting audio from {video_path}: {e}")
        return None

def load_audio(audio_path, target_sr=16000):
    """Load audio file and convert to target sample rate."""
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)
        
        return waveform.squeeze().numpy(), target_sr
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None, None

def extract_wav2vec_features(audio_path, processor, model, device):
    """Extract wav2vec 2.0 features from audio file."""
    try:
        # Load audio
        waveform, sample_rate = load_audio(audio_path)
        if waveform is None:
            return None
        
        # Process audio with wav2vec
        inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Extract features
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the hidden states from the last layer
            last_hidden = outputs.last_hidden_state
            
            # Convert to numpy
            features = last_hidden.cpu().numpy().squeeze()
            
            return features
    except Exception as e:
        print(f"Error extracting wav2vec features from {audio_path}: {e}")
        return None

def process_ravdess_file(video_path, output_dir, processor, model, device, temp_dir=None):
    """Process a single RAVDESS video file."""
    
    try:
        # Extract emotion code from filename
        # Format: 01-01-03-01-01-01-01.mp4
        # where 3rd part (03) is the emotion code
        filename = os.path.basename(video_path)
        parts = filename.split('-')
        
        if len(parts) < 3:
            print(f"Invalid filename format: {filename}")
            return None
        
        emotion_code = parts[2]
        if emotion_code not in RAVDESS_EMOTION_MAP:
            print(f"Unknown emotion code: {emotion_code}")
            return None
        
        emotion = RAVDESS_EMOTION_MAP[emotion_code]
        emotion_class = EMOTION_CLASS_MAP.get(emotion)
        if emotion_class is None:
            print(f"Emotion {emotion} not in class map, skipping")
            return None
        
        # Create temp audio directory if needed
        if temp_dir:
            os.makedirs(temp_dir, exist_ok=True)
            temp_audio_path = os.path.join(temp_dir, f"{os.path.splitext(filename)[0]}.wav")
        else:
            temp_audio_path = None
        
        # Extract audio from video
        audio_path = extract_audio_with_ffmpeg(video_path, temp_audio_path)
        if audio_path is None:
            return None
        
        # Extract wav2vec features
        features = extract_wav2vec_features(audio_path, processor, model, device)
        
        # Remove temporary audio file if it wasn't specified by the user
        if temp_dir is None and os.path.exists(audio_path):
            os.remove(audio_path)
            
        if features is None:
            return None
        
        # Create one-hot encoded label
        label = np.zeros(6)  # 6 classes: angry, disgust, fearful, happy, neutral, sad
        label[emotion_class] = 1
        
        # Save features to NPZ file
        output_file = os.path.join(output_dir, f"ravdess_{filename.replace('.mp4', '')}.npz")
        np.savez(
            output_file, 
            wav2vec_features=features, 
            label=label, 
            emotion=emotion,
            emotion_class=emotion_class
        )
        
        return output_file
    except Exception as e:
        print(f"Error processing RAVDESS file {video_path}: {e}")
        return None

def process_cremad_file(video_path, output_dir, processor, model, device, temp_dir=None):
    """Process a single CREMA-D video file."""
    
    try:
        # Extract emotion from filename
        # Typical CREMA-D format: 1076_MTI_SAD_XX.mp4 or 1076_MTI_SAD_XX.flv
        # where the 3rd part is the emotion
        filename = os.path.basename(video_path)
        
        # Check if it's a CREMA-D file with correct naming pattern
        if '_' not in filename:
            return None  # Not a CREMA-D format file
            
        parts = filename.split('_')
        if len(parts) < 3:
            print(f"Invalid CREMA-D filename format: {filename}")
            return None
        
        emotion = parts[2].lower()
        
        # Map CREMA-D emotion names to our standard format
        emotion_mapping = {
            'ang': 'angry',
            'dis': 'disgust',
            'fea': 'fearful',
            'hap': 'happy',
            'neu': 'neutral',
            'sad': 'sad'
        }
        
        # Convert to standard emotion name if needed
        if emotion in emotion_mapping:
            emotion = emotion_mapping[emotion]
        
        emotion_class = EMOTION_CLASS_MAP.get(emotion)
        if emotion_class is None:
            print(f"Emotion {emotion} not in class map, skipping")
            return None
        
        # Create temp audio directory if needed
        if temp_dir:
            os.makedirs(temp_dir, exist_ok=True)
            temp_audio_path = os.path.join(temp_dir, f"{os.path.splitext(filename)[0]}.wav")
        else:
            temp_audio_path = None
        
        # Extract audio from video
        audio_path = extract_audio_with_ffmpeg(video_path, temp_audio_path)
        if audio_path is None:
            return None
        
        # Extract wav2vec features
        features = extract_wav2vec_features(audio_path, processor, model, device)
        
        # Remove temporary audio file if it wasn't specified by the user
        if temp_dir is None and os.path.exists(audio_path):
            os.remove(audio_path)
            
        if features is None:
            return None
        
        # Create one-hot encoded label
        label = np.zeros(6)  # 6 classes
        label[emotion_class] = 1
        
        # Get the extension from the original file
        extension = os.path.splitext(filename)[1]
        # Save features to NPZ file, preserving the original extension in the filename
        output_file = os.path.join(output_dir, f"cremad_{os.path.splitext(filename)[0]}.npz")
        np.savez(
            output_file, 
            wav2vec_features=features, 
            label=label, 
            emotion=emotion,
            emotion_class=emotion_class
        )
        
        return output_file
    except Exception as e:
        print(f"Error processing CREMA-D file {video_path}: {e}")
        return None

def create_fusion_model(video_model_path, audio_model_config, video_weight=0.7, audio_weight=0.3):
    """Create and save a fusion model configuration."""
    config = {
        "fusion_type": "late",
        "video_model_path": video_model_path,
        "audio_model_config": audio_model_config,
        "video_weight": video_weight,
        "audio_weight": audio_weight,
        "emotion_labels": ["angry", "disgust", "fearful", "happy", "neutral", "sad"]
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(audio_model_config['output_path']), exist_ok=True)
    
    # Save config
    config_path = os.path.join(os.path.dirname(audio_model_config['output_path']), 'fusion_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Fusion model configuration saved to {config_path}")
    return config_path

def main():
    parser = argparse.ArgumentParser(description="Extract wav2vec features and set up fusion model")
    parser.add_argument("--ravdess_dir", type=str, default="/home/ubuntu/datasets/ravdess_videos",
                      help="Path to RAVDESS dataset directory")
    parser.add_argument("--cremad_dir", type=str, default="/home/ubuntu/datasets/crema_d_videos",
                      help="Path to CREMA-D dataset directory")
    parser.add_argument("--output_dir", type=str, default="models/wav2vec",
                      help="Directory to save wav2vec features")
    parser.add_argument("--temp_dir", type=str, default=None,
                      help="Directory to save temporary audio files (None for system temp)")
    parser.add_argument("--video_weight", type=float, default=0.7,
                      help="Weight for video model in fusion (0.0-1.0)")
    parser.add_argument("--audio_weight", type=float, default=0.3,
                      help="Weight for audio model in fusion (0.0-1.0)")
    parser.add_argument("--num_samples", type=int, default=20,
                      help="Number of samples to process from each dataset (for testing)")
    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-base-960h",
                      help="Name of wav2vec model to use")
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create temp audio directory if specified
    if args.temp_dir:
        os.makedirs(args.temp_dir, exist_ok=True)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load wav2vec model
    print("Loading wav2vec model...")
    processor = Wav2Vec2Processor.from_pretrained(args.model_name)
    model = Wav2Vec2Model.from_pretrained(args.model_name).to(device)
    model.eval()
    
    # Process RAVDESS files
    print(f"Searching for RAVDESS files in {args.ravdess_dir}...")
    ravdess_files = glob.glob(os.path.join(args.ravdess_dir, "**/*.mp4"), recursive=True)
    
    if not ravdess_files:
        print(f"No MP4 files found in {args.ravdess_dir}")
    else:
        print(f"Found {len(ravdess_files)} RAVDESS files")
        
        # If num_samples is specified, sample that many files
        if args.num_samples and args.num_samples < len(ravdess_files):
            ravdess_files = random.sample(ravdess_files, args.num_samples)
            print(f"Sampled {args.num_samples} RAVDESS files for processing")
        
        # Process files sequentially (could be parallelized for production)
        ravdess_outputs = []
        for file in tqdm(ravdess_files, desc="Processing RAVDESS"):
            result = process_ravdess_file(file, args.output_dir, processor, model, device, args.temp_dir)
            if result:
                ravdess_outputs.append(result)
        
        print(f"Successfully processed {len(ravdess_outputs)} RAVDESS files")
    
    # Process CREMA-D files - Look for both MP4 and FLV files
    print(f"Searching for CREMA-D files in {args.cremad_dir}...")
    cremad_mp4_files = glob.glob(os.path.join(args.cremad_dir, "**/*.mp4"), recursive=True)
    cremad_flv_files = glob.glob(os.path.join(args.cremad_dir, "**/*.flv"), recursive=True)
    cremad_files = cremad_mp4_files + cremad_flv_files
    
    if not cremad_files:
        # If still no files found, try to list all files to see what's there
        print(f"No MP4 or FLV files found in {args.cremad_dir}")
        print("Checking all files in the directory:")
        all_files = glob.glob(os.path.join(args.cremad_dir, "**/*.*"), recursive=True)
        if all_files:
            print(f"Found {len(all_files)} files with the following extensions:")
            extensions = set([os.path.splitext(f)[1] for f in all_files])
            for ext in extensions:
                count = sum(1 for f in all_files if f.endswith(ext))
                print(f"  {ext}: {count} files")
            
            # Try to process all video files regardless of extension
            cremad_files = [f for f in all_files if os.path.splitext(f)[1].lower() in 
                          ['.mp4', '.flv', '.avi', '.mov', '.wmv', '.mkv']]
            print(f"Will attempt to process {len(cremad_files)} video files")
    
    if cremad_files:
        print(f"Found {len(cremad_files)} CREMA-D files")
        
        # If num_samples is specified, sample that many files
        if args.num_samples and args.num_samples < len(cremad_files):
            cremad_files = random.sample(cremad_files, args.num_samples)
            print(f"Sampled {args.num_samples} CREMA-D files for processing")
        
        # Process files sequentially
        cremad_outputs = []
        for file in tqdm(cremad_files, desc="Processing CREMA-D"):
            result = process_cremad_file(file, args.output_dir, processor, model, device, args.temp_dir)
            if result:
                cremad_outputs.append(result)
        
        print(f"Successfully processed {len(cremad_outputs)} CREMA-D files")
    
    # Create fusion model configuration
    total_processed = len(ravdess_outputs) + len(cremad_outputs) if 'ravdess_outputs' in locals() and 'cremad_outputs' in locals() else 0
    if total_processed > 0:
        print("\nSetting up fusion model...")
        
        audio_model_config = {
            "features_path": args.output_dir,
            "output_path": os.path.join("models/fusion", "audio_model.h5"),
            "num_features": 768  # wav2vec base hidden size
        }
        
        fusion_config_path = create_fusion_model(
            video_model_path="models/slowfast_emotion_video_only_92.9.pt",
            audio_model_config=audio_model_config,
            video_weight=args.video_weight,
            audio_weight=args.audio_weight
        )
        
        print("\nDone!")
        print(f"Extracted wav2vec features for {total_processed} files")
        print(f"Fusion model configuration saved to {fusion_config_path}")
        print("\nYou can now use these features to train a wav2vec-based audio model")
        print("And then combine it with the SlowFast video model using the fusion configuration")
    else:
        print("\nNo files were processed successfully. Please check the dataset paths.")

if __name__ == "__main__":
    main()
