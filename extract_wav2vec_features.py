#!/usr/bin/env python3
"""
Extract wav2vec features from RAVDESS and CREMA-D audio for emotion recognition.
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

def get_ravdess_emotion(filename):
    """Extract emotion from RAVDESS filename."""
    parts = filename.split('-')
    if len(parts) < 3:
        return None
    
    emotion_code = parts[2]
    emotion_map = {
        '01': 'neutral', '02': 'neutral', 
        '03': 'happy', '04': 'sad', 
        '05': 'angry', '06': 'fearful', 
        '07': 'disgust'
    }
    return emotion_map.get(emotion_code, None)

def get_cremad_emotion(filename):
    """Extract emotion from CREMA-D filename."""
    parts = filename.split('_')
    if len(parts) < 3:
        return None
    
    emotion_code = parts[2].upper()
    emotion_map = {
        'ANG': 'angry', 'DIS': 'disgust',
        'FEA': 'fearful', 'HAP': 'happy',
        'NEU': 'neutral', 'SAD': 'sad'
    }
    return emotion_map.get(emotion_code, None)

def main():
    # Setup
    os.makedirs("models/wav2vec", exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading wav2vec model...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    model.eval()
    
    # Find audio files
    ravdess_pattern = "downsampled_videos/RAVDESS/*/*/*.wav"
    cremad_pattern = "downsampled_videos/CREMA-D-audio-complete/*.wav"
    
    ravdess_files = glob.glob(ravdess_pattern)[:10]  # Process just a few for testing
    cremad_files = glob.glob(cremad_pattern)[:10]
    
    print(f"Found {len(ravdess_files)} RAVDESS files")
    print(f"Found {len(cremad_files)} CREMA-D files")
    
    # Process RAVDESS
    for file_path in tqdm(ravdess_files, desc="Processing RAVDESS"):
        try:
            filename = os.path.basename(file_path)
            emotion = get_ravdess_emotion(filename)
            if emotion is None or emotion not in EMOTION_MAP:
                continue
            
            # Extract features
            features = extract_wav2vec_features(file_path, processor, model, device)
            if features is None:
                continue
            
            # Save to NPZ
            output_file = os.path.join("models/wav2vec", f"ravdess_{filename.replace('.wav', '')}.npz")
            np.savez(
                output_file,
                wav2vec_features=features,
                emotion=emotion,
                label=EMOTION_MAP[emotion]
            )
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Process CREMA-D
    for file_path in tqdm(cremad_files, desc="Processing CREMA-D"):
        try:
            filename = os.path.basename(file_path)
            emotion = get_cremad_emotion(filename)
            if emotion is None or emotion not in EMOTION_MAP:
                continue
            
            # Extract features
            features = extract_wav2vec_features(file_path, processor, model, device)
            if features is None:
                continue
            
            # Save to NPZ
            output_file = os.path.join("models/wav2vec", f"cremad_{filename.replace('.wav', '')}.npz")
            np.savez(
                output_file,
                wav2vec_features=features,
                emotion=emotion,
                label=EMOTION_MAP[emotion]
            )
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print("Feature extraction complete!")
    print(f"Features saved to models/wav2vec/")

if __name__ == "__main__":
    main()
