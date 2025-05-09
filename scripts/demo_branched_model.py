#!/usr/bin/env python3
"""
Demo script to show the trained branched model for emotion recognition on sample data.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Define paths
MODEL_PATH = "models/branched_6class/final_model.h5"
RAVDESS_DIR = "ravdess_features"
CREMA_D_DIR = "crema_d_features"

# Define emotion classes
EMOTION_CLASSES = [
    'Neutral/Calm', 'Happy', 'Sad', 
    'Angry', 'Fearful', 'Disgust'
]

def load_sample_data(ravdess_file, crema_d_file):
    """Load sample data from RAVDESS and CREMA-D datasets"""
    samples = []
    
    # Load RAVDESS sample
    if os.path.exists(ravdess_file):
        ravdess_data = np.load(ravdess_file, allow_pickle=True)
        samples.append({
            'name': f"RAVDESS: {os.path.basename(ravdess_file)}",
            'video_features': ravdess_data['video_features'],
            'audio_features': ravdess_data['audio_features'],
            'emotion_label': ravdess_data['emotion_label'].item() if 'emotion_label' in ravdess_data else None,
            'dataset': 'RAVDESS'
        })
        print(f"Loaded RAVDESS sample: {os.path.basename(ravdess_file)}")
    
    # Load CREMA-D sample
    if os.path.exists(crema_d_file):
        crema_d_data = np.load(crema_d_file, allow_pickle=True)
        samples.append({
            'name': f"CREMA-D: {os.path.basename(crema_d_file)}",
            'video_features': crema_d_data['video_features'],
            'audio_features': crema_d_data['audio_features'],
            'emotion_label': crema_d_data['emotion_label'].item() if 'emotion_label' in crema_d_data else None,
            'dataset': 'CREMA-D'
        })
        print(f"Loaded CREMA-D sample: {os.path.basename(crema_d_file)}")
    
    return samples

def pad_sequences(video_features, audio_features, max_video_length=68, max_audio_length=447):
    """Pad sequences to the specified maximum lengths"""
    # Pad video features
    video_length = min(video_features.shape[0], max_video_length)
    padded_video = np.zeros((max_video_length, video_features.shape[1]))
    padded_video[:video_length] = video_features[:video_length]
    
    # Pad audio features
    audio_length = min(audio_features.shape[0], max_audio_length)
    padded_audio = np.zeros((max_audio_length, audio_features.shape[1]))
    padded_audio[:audio_length] = audio_features[:audio_length]
    
    return padded_video, padded_audio

def predict_emotion(model, samples, max_video_length=68, max_audio_length=447):
    """Use the model to predict emotions for the samples"""
    results = []
    
    for sample in samples:
        # Pad sequences
        padded_video, padded_audio = pad_sequences(
            sample['video_features'], 
            sample['audio_features'],
            max_video_length,
            max_audio_length
        )
        
        # Expand dimensions for batch size of 1
        padded_video = np.expand_dims(padded_video, axis=0)
        padded_audio = np.expand_dims(padded_audio, axis=0)
        
        # Make prediction
        prediction = model.predict([padded_video, padded_audio], verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        # Map true label
        if sample['dataset'] == 'RAVDESS':
            # RAVDESS: Remap emotion (0=neutral, 1=calm -> 0=neutral/calm)
            true_label = sample['emotion_label']
            if true_label == 0 or true_label == 1:
                true_emotion = 0  # Neutral/Calm
            elif true_label >= 2 and true_label <= 6:
                true_emotion = true_label - 1  # Adjust for merged neutral/calm
            else:
                true_emotion = None  # Unknown or surprised (not in 6-class)
        else:
            # CREMA-D: Direct mapping
            true_emotion = sample['emotion_label']
        
        # Store results
        results.append({
            'name': sample['name'],
            'prediction': predicted_class,
            'prediction_name': EMOTION_CLASSES[predicted_class],
            'confidence': confidence,
            'true_label': true_emotion,
            'true_name': EMOTION_CLASSES[true_emotion] if true_emotion is not None and 0 <= true_emotion < len(EMOTION_CLASSES) else "Unknown",
            'correct': predicted_class == true_emotion
        })
    
    return results

def display_results(results):
    """Display the prediction results"""
    print("\nPrediction Results:")
    print("=" * 80)
    
    for result in results:
        print(f"Sample: {result['name']}")
        print(f"True emotion: {result['true_name']} (class {result['true_label']})")
        print(f"Predicted: {result['prediction_name']} (class {result['prediction']}) with {result['confidence']:.4f} confidence")
        print(f"Correct: {'Yes' if result['correct'] else 'No'}")
        print("-" * 80)
    
    # Calculate overall accuracy
    correct = sum(1 for r in results if r['correct'])
    print(f"Overall accuracy: {correct}/{len(results)} ({correct/len(results)*100:.2f}%)")

def plot_emotion_distribution(samples):
    """Plot emotion distribution of all samples in the dataset"""
    # Count emotions in RAVDESS
    ravdess_dir = RAVDESS_DIR
    ravdess_emotions = []
    
    if os.path.exists(ravdess_dir):
        for file in os.listdir(ravdess_dir):
            if file.endswith('.npz'):
                try:
                    data = np.load(os.path.join(ravdess_dir, file), allow_pickle=True)
                    if 'emotion_label' in data:
                        emotion = data['emotion_label'].item()
                        # Remap 0,1 -> 0, 2->1, etc.
                        if emotion == 0 or emotion == 1:
                            emotion = 0
                        elif emotion >= 2 and emotion <= 6:
                            emotion = emotion - 1
                        else:
                            continue  # Skip surprised or unknown emotions
                        ravdess_emotions.append(emotion)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
    
    # Count emotions in CREMA-D
    crema_d_dir = CREMA_D_DIR
    crema_d_emotions = []
    
    if os.path.exists(crema_d_dir):
        for file in os.listdir(crema_d_dir)[:100]:  # Limit to 100 files for speed
            if file.endswith('.npz'):
                try:
                    data = np.load(os.path.join(crema_d_dir, file), allow_pickle=True)
                    if 'emotion_label' in data:
                        emotion = data['emotion_label'].item()
                        crema_d_emotions.append(emotion)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot RAVDESS
    if ravdess_emotions:
        plt.subplot(1, 2, 1)
        ravdess_counts = np.bincount(ravdess_emotions)
        plt.bar(range(len(ravdess_counts)), ravdess_counts, tick_label=[EMOTION_CLASSES[i] if i < len(EMOTION_CLASSES) else f"Unknown({i})" for i in range(len(ravdess_counts))])
        plt.title('RAVDESS Emotion Distribution')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
    
    # Plot CREMA-D
    if crema_d_emotions:
        plt.subplot(1, 2, 2)
        crema_d_counts = np.bincount(crema_d_emotions)
        plt.bar(range(len(crema_d_counts)), crema_d_counts, tick_label=[EMOTION_CLASSES[i] if i < len(EMOTION_CLASSES) else f"Unknown({i})" for i in range(len(crema_d_counts))])
        plt.title('CREMA-D Emotion Distribution (Sample)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
    
    plt.savefig('emotion_distribution.png')
    print("Emotion distribution plot saved to emotion_distribution.png")

def main():
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return
    
    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    model.summary()
    
    # Define samples to test
    ravdess_sample = os.path.join(RAVDESS_DIR, "01-01-04-01-01-01-01.npz")  # Sad emotion
    crema_d_sample = os.path.join(CREMA_D_DIR, "1001_DFA_ANG_XX.npz")      # Angry emotion
    
    # Load samples
    samples = load_sample_data(ravdess_sample, crema_d_sample)
    
    if not samples:
        print("Error: No samples could be loaded")
        return
    
    # Make predictions
    results = predict_emotion(model, samples)
    
    # Display results
    display_results(results)
    
    # Plot emotion distribution (optional)
    # Uncomment to generate emotion distribution plot
    # plot_emotion_distribution(samples)

if __name__ == "__main__":
    main()
