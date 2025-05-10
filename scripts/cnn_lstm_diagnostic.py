#!/usr/bin/env python3
"""
Diagnostic script to investigate why the CNN-LSTM model isn't learning effectively.
This analyzes class distribution, feature quality, and label mapping issues.
"""

import os
import sys
import numpy as np
import glob
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import plot_model

# Directory paths
RAVDESS_CNN_AUDIO_DIR = "data/ravdess_features_cnn_fixed" 
CREMA_D_CNN_AUDIO_DIR = "data/crema_d_features_cnn_fixed"

# Emotion maps
EMOTION_MAP = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}
EMOTION_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']
RAVDESS_EMOTION_MAP = {'01': 'NEU', '02': 'NEU', '03': 'HAP', '04': 'SAD', '05': 'ANG', '06': 'FEA', '07': 'DIS'}

# Helper function to extract label from filename
def extract_label_from_filename(filename):
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    if "Actor_" in filename:  # RAVDESS
        parts = base_name.split('-')
        try:
            emotion_code = RAVDESS_EMOTION_MAP.get(parts[2], None)
            return emotion_code, EMOTION_MAP.get(emotion_code, None) if emotion_code else None
        except:
            return None, None
    else:  # CREMA-D
        parts = base_name.split('_')
        try:
            emotion_code = parts[2]
            return emotion_code, EMOTION_MAP.get(emotion_code, None)
        except:
            return None, None

# Helper function to load CNN audio features
def load_cnn_audio_feature(file_path):
    try:
        feature = np.load(file_path)
        return feature
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def analyze_class_distribution():
    """Analyze class distribution in both datasets and plot bar chart."""
    print("="*50)
    print("ANALYZING CLASS DISTRIBUTION")
    print("="*50)
    
    ravdess_files = glob.glob(os.path.join(RAVDESS_CNN_AUDIO_DIR, "Actor_*", "*.npy"))
    cremad_files = glob.glob(os.path.join(CREMA_D_CNN_AUDIO_DIR, "*.npy"))
    
    print(f"RAVDESS files: {len(ravdess_files)}")
    print(f"CREMA-D files: {len(cremad_files)}")
    
    # Count emotions
    ravdess_emotions = []
    cremad_emotions = []
    
    for file in tqdm(ravdess_files, desc="Processing RAVDESS"):
        emotion_code, emotion_idx = extract_label_from_filename(file)
        if emotion_idx is not None:
            ravdess_emotions.append(emotion_idx)
    
    for file in tqdm(cremad_files, desc="Processing CREMA-D"):
        emotion_code, emotion_idx = extract_label_from_filename(file)
        if emotion_idx is not None:
            cremad_emotions.append(emotion_idx)
    
    print(f"RAVDESS valid labels: {len(ravdess_emotions)}")
    print(f"CREMA-D valid labels: {len(cremad_emotions)}")
    
    # Count frequencies
    ravdess_counts = Counter(ravdess_emotions)
    cremad_counts = Counter(cremad_emotions)
    
    print("\nRAVDESS Emotion Distribution:")
    for emotion_idx in range(6):
        count = ravdess_counts.get(emotion_idx, 0)
        percentage = 100 * count / len(ravdess_emotions) if ravdess_emotions else 0
        print(f"{EMOTION_NAMES[emotion_idx]}: {count} ({percentage:.1f}%)")
    
    print("\nCREMA-D Emotion Distribution:")
    for emotion_idx in range(6):
        count = cremad_counts.get(emotion_idx, 0)
        percentage = 100 * count / len(cremad_emotions) if cremad_emotions else 0
        print(f"{EMOTION_NAMES[emotion_idx]}: {count} ({percentage:.1f}%)")
    
    # Combine for total
    combined_emotions = ravdess_emotions + cremad_emotions
    combined_counts = Counter(combined_emotions)
    
    print("\nCOMBINED Dataset Emotion Distribution:")
    for emotion_idx in range(6):
        count = combined_counts.get(emotion_idx, 0)
        percentage = 100 * count / len(combined_emotions) if combined_emotions else 0
        print(f"{EMOTION_NAMES[emotion_idx]}: {count} ({percentage:.1f}%)")
    
    # Check for train/val balance issues
    print("\nEvaluating potential train/val split imbalance...")
    
    # Simulate the train/val split (80/20)
    from sklearn.model_selection import train_test_split
    
    # Create one-hot encoded labels
    labels = np.zeros((len(combined_emotions), 6))
    for i, emotion_idx in enumerate(combined_emotions):
        labels[i, emotion_idx] = 1
    
    # Split data
    train_idx, val_idx = train_test_split(
        np.arange(len(combined_emotions)), 
        test_size=0.2, 
        random_state=42,
        stratify=labels  # Ensure balanced split
    )
    
    train_emotions = [combined_emotions[i] for i in train_idx]
    val_emotions = [combined_emotions[i] for i in val_idx]
    
    train_counts = Counter(train_emotions)
    val_counts = Counter(val_emotions)
    
    print("\nTrain Split Emotion Distribution:")
    for emotion_idx in range(6):
        count = train_counts.get(emotion_idx, 0)
        percentage = 100 * count / len(train_emotions) if train_emotions else 0
        print(f"{EMOTION_NAMES[emotion_idx]}: {count} ({percentage:.1f}%)")
    
    print("\nValidation Split Emotion Distribution:")
    for emotion_idx in range(6):
        count = val_counts.get(emotion_idx, 0)
        percentage = 100 * count / len(val_emotions) if val_emotions else 0
        print(f"{EMOTION_NAMES[emotion_idx]}: {count} ({percentage:.1f}%)")
    
    # Plot distribution
    plt.figure(figsize=(15, 10))
    
    # Plot RAVDESS
    plt.subplot(3, 1, 1)
    counts = [ravdess_counts.get(i, 0) for i in range(6)]
    plt.bar(EMOTION_NAMES, counts)
    plt.title("RAVDESS Emotion Distribution")
    plt.ylabel("Count")
    
    # Plot CREMA-D
    plt.subplot(3, 1, 2)
    counts = [cremad_counts.get(i, 0) for i in range(6)]
    plt.bar(EMOTION_NAMES, counts)
    plt.title("CREMA-D Emotion Distribution")
    plt.ylabel("Count")
    
    # Plot combined
    plt.subplot(3, 1, 3)
    counts = [combined_counts.get(i, 0) for i in range(6)]
    plt.bar(EMOTION_NAMES, counts)
    plt.title("Combined Dataset Emotion Distribution")
    plt.ylabel("Count")
    
    plt.tight_layout()
    plt.savefig("emotion_distribution.png")
    print(f"Distribution plot saved to emotion_distribution.png")

def analyze_feature_quality():
    """Analyze feature quality and dimensions."""
    print("="*50)
    print("ANALYZING FEATURE QUALITY")
    print("="*50)
    
    # Sample a few files from each dataset
    ravdess_files = glob.glob(os.path.join(RAVDESS_CNN_AUDIO_DIR, "Actor_*", "*.npy"))
    cremad_files = glob.glob(os.path.join(CREMA_D_CNN_AUDIO_DIR, "*.npy"))
    
    np.random.seed(42)
    ravdess_samples = np.random.choice(ravdess_files, min(10, len(ravdess_files)), replace=False)
    cremad_samples = np.random.choice(cremad_files, min(10, len(cremad_files)), replace=False)
    
    # Analyze RAVDESS samples
    print("\nRAVDESS Feature Analysis:")
    ravdess_shapes = []
    ravdess_means = []
    ravdess_stds = []
    ravdess_mins = []
    ravdess_maxs = []
    
    for file in ravdess_samples:
        feature = load_cnn_audio_feature(file)
        if feature is not None:
            ravdess_shapes.append(feature.shape)
            ravdess_means.append(np.mean(feature))
            ravdess_stds.append(np.std(feature))
            ravdess_mins.append(np.min(feature))
            ravdess_maxs.append(np.max(feature))
            print(f"File: {os.path.basename(file)}")
            print(f"  Shape: {feature.shape}")
            print(f"  Mean: {np.mean(feature):.4f}, Std: {np.std(feature):.4f}")
            print(f"  Min: {np.min(feature):.4f}, Max: {np.max(feature):.4f}")
    
    # Analyze CREMA-D samples
    print("\nCREMA-D Feature Analysis:")
    cremad_shapes = []
    cremad_means = []
    cremad_stds = []
    cremad_mins = []
    cremad_maxs = []
    
    for file in cremad_samples:
        feature = load_cnn_audio_feature(file)
        if feature is not None:
            cremad_shapes.append(feature.shape)
            cremad_means.append(np.mean(feature))
            cremad_stds.append(np.std(feature))
            cremad_mins.append(np.min(feature))
            cremad_maxs.append(np.max(feature))
            print(f"File: {os.path.basename(file)}")
            print(f"  Shape: {feature.shape}")
            print(f"  Mean: {np.mean(feature):.4f}, Std: {np.std(feature):.4f}")
            print(f"  Min: {np.min(feature):.4f}, Max: {np.max(feature):.4f}")
    
    # Check for consistency
    print("\nFeature Shape Consistency Check:")
    if len(set(s[1] for s in ravdess_shapes)) == 1 and len(set(s[1] for s in cremad_shapes)) == 1:
        ravdess_dim = ravdess_shapes[0][1] if ravdess_shapes else None
        cremad_dim = cremad_shapes[0][1] if cremad_shapes else None
        if ravdess_dim == cremad_dim:
            print(f"✅ Consistent feature dimension across datasets: {ravdess_dim}")
        else:
            print(f"❌ Inconsistent feature dimensions: RAVDESS={ravdess_dim}, CREMA-D={cremad_dim}")
    else:
        print("❌ Inconsistent feature dimensions within datasets!")
        print(f"RAVDESS shapes: {set(s[1] for s in ravdess_shapes)}")
        print(f"CREMA-D shapes: {set(s[1] for s in cremad_shapes)}")
    
    # Check for file naming consistency with label extraction
    print("\nLabel Extraction Consistency Check:")
    all_files = ravdess_files + cremad_files
    sample_files = np.random.choice(all_files, min(20, len(all_files)), replace=False)
    
    extraction_success = 0
    for file in sample_files:
        emotion_code, emotion_idx = extract_label_from_filename(file)
        if emotion_idx is not None:
            extraction_success += 1
            print(f"File: {os.path.basename(file)}")
            print(f"  Extracted: Emotion Code={emotion_code}, Index={emotion_idx} ({EMOTION_NAMES[emotion_idx]})")
    
    success_rate = 100 * extraction_success / len(sample_files) if len(sample_files) > 0 else 0
    print(f"\nLabel extraction success rate: {success_rate:.1f}%")
    if success_rate < 95:
        print("❌ Low label extraction success rate! Check file naming patterns.")
    else:
        print("✅ Good label extraction rate")

def create_test_model():
    """Create a simplified test model for quick verification."""
    print("="*50)
    print("CREATING TEST MODEL")
    print("="*50)
    
    # Determine input dimension from a sample file
    sample_files = glob.glob(os.path.join(RAVDESS_CNN_AUDIO_DIR, "Actor_*", "*.npy"))
    if not sample_files:
        sample_files = glob.glob(os.path.join(CREMA_D_CNN_AUDIO_DIR, "*.npy"))
    
    if not sample_files:
        print("❌ No sample files found to determine input dimension!")
        return
    
    sample_feature = load_cnn_audio_feature(sample_files[0])
    if sample_feature is None:
        print("❌ Failed to load sample feature for model test!")
        return
    
    input_dim = sample_feature.shape[1]
    print(f"Using input dimension: {input_dim}")
    
    # Create a simple model for testing
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    
    model = Sequential([
        LSTM(64, input_shape=(None, input_dim)),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(6, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Try a very simple overfit test with a few samples
    print("\nAttempting simple overfit test...")
    
    # Collect a few samples
    all_files = glob.glob(os.path.join(RAVDESS_CNN_AUDIO_DIR, "Actor_*", "*.npy")) + \
               glob.glob(os.path.join(CREMA_D_CNN_AUDIO_DIR, "*.npy"))
    
    # Get 10 samples with valid labels
    test_samples = []
    test_labels = []
    
    for file in all_files:
        if len(test_samples) >= 10:
            break
        
        emotion_code, emotion_idx = extract_label_from_filename(file)
        if emotion_idx is not None:
            feature = load_cnn_audio_feature(file)
            if feature is not None:
                test_samples.append(feature)
                label = np.zeros(6)
                label[emotion_idx] = 1
                test_labels.append(label)
    
    if not test_samples:
        print("❌ No valid test samples found for overfit test!")
        return
    
    # Convert to arrays
    X = np.array(test_samples)
    y = np.array(test_labels)
    
    print(f"Test samples: {X.shape}")
    print(f"Test labels: {y.shape}")
    
    # Try to overfit
    history = model.fit(
        X, y,
        epochs=50,
        batch_size=5,
        verbose=1
    )
    
    # Plot training results
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'])
    plt.title('Model Overfit Test Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.savefig("overfit_test.png")
    print(f"Overfit test plot saved to overfit_test.png")
    
    # Check final accuracy
    final_acc = history.history['accuracy'][-1]
    print(f"Final overfit accuracy: {final_acc:.4f}")
    
    if final_acc > 0.8:
        print("✅ Overfit test successful! The model can learn the data pattern.")
    else:
        print("❌ Overfit test failed! The model struggles to learn even a small sample.")

def recommend_fixes():
    """Provide recommendations based on diagnostic results."""
    print("="*50)
    print("RECOMMENDATIONS")
    print("="*50)
    
    print("""
Based on the diagnostic results, here are some possible fixes:

1. For class imbalance:
   - Apply class weights in model.fit
   - Use stratified sampling in data generators
   - Consider data augmentation for underrepresented classes

2. For feature issues:
   - Verify CNN feature extraction process
   - Check for normalization issues (outliers, inconsistent ranges)
   - Consider dimensionality reduction (PCA)

3. For model architecture:
   - Increase model capacity (more LSTM units or layers)
   - Try bidirectional LSTMs
   - Reduce L2 regularization if features are sparse
   - Increase learning rate if training is too slow

4. For overfitting:
   - Increase dropout
   - Apply recurrent dropout
   - Add batch normalization
   - Reduce model capacity

5. For implementation issues:
   - Check label extraction logic
   - Verify data loading/batch generation is correct
   - Monitor learning rate schedule
   - Use TensorBoard to track training metrics
    """)

if __name__ == "__main__":
    print("CNN-LSTM Model Diagnostic Tool")
    print("=" * 50)
    
    # Check TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    print(f"NumPy version: {np.__version__}")
    print("-" * 50)
    
    # Run diagnostics
    analyze_class_distribution()
    analyze_feature_quality()
    create_test_model()
    recommend_fixes()
    
    print("\nDiagnostic complete!")
