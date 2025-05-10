#!/usr/bin/env python3
"""
Diagnostic script to find the root cause of random accuracy issues.
Performs two tests:
1. Overfitting test - Can the model learn from a small subset of data?
2. Feature baseline test - Can LogisticRegression classify time-averaged features?
"""

import os
import sys
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# Directory settings - same as in existing scripts
RAVDESS_CNN_AUDIO_DIR = "data/ravdess_features_cnn_fixed"
CREMA_D_CNN_AUDIO_DIR = "data/crema_d_features_cnn_fixed"

# Constants
NUM_CLASSES = 6
SAMPLES_PER_CLASS = 10  # For overfitting test
RANDOM_SEED = 42

print("CNN FEATURE DIAGNOSTIC")
print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)
print("Python version:", sys.version)


def load_data_paths_and_labels_audio_only(cnn_audio_dir_ravdess, cnn_audio_dir_cremad):
    """Load file paths and extract labels - copied from training script."""
    cnn_audio_files = glob.glob(os.path.join(cnn_audio_dir_ravdess, "*", "*.npy")) + \
                      glob.glob(os.path.join(cnn_audio_dir_cremad, "*.npy"))

    labels = []
    emotion_map = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}
    ravdess_emotion_map = {'01': 'NEU', '02': 'NEU', '03': 'HAP', '04': 'SAD', '05': 'ANG', '06': 'FEA', '07': 'DIS'}

    print(f"Found {len(cnn_audio_files)} CNN audio files. Extracting labels...")
    skipped_label = 0
    valid_cnn_audio_files = []
    mapped_labels = []  # Extra list to keep track of actual emotion codes for debugging

    for cnn_audio_file in tqdm(cnn_audio_files, desc="Extracting labels"):
        base_name = os.path.splitext(os.path.basename(cnn_audio_file))[0]
        label = None
        emotion_code = None

        # Extract label
        try:
            if "Actor_" in cnn_audio_file:  # RAVDESS
                parts = base_name.split('-')
                emotion_code = ravdess_emotion_map.get(parts[2], None)
                if emotion_code in emotion_map:
                    label = np.zeros(len(emotion_map))
                    label[emotion_map[emotion_code]] = 1
            else:  # CREMA-D
                parts = base_name.split('_')
                emotion_code = parts[2]
                if emotion_code in emotion_map:
                    label = np.zeros(len(emotion_map))
                    label[emotion_map[emotion_code]] = 1
        except Exception as e:
            print(f"Label parsing error for {cnn_audio_file}: {e}")
            label = None

        if label is not None:
            valid_cnn_audio_files.append(cnn_audio_file)
            labels.append(label)
            mapped_labels.append(emotion_code)
        else:
            skipped_label += 1

    print(f"Found {len(valid_cnn_audio_files)} CNN audio files with valid labels.")
    print(f"Skipped {skipped_label} due to label parsing issues.")

    if not valid_cnn_audio_files:
        raise FileNotFoundError("No CNN audio files with valid labels found.")
        
    # Print emotion distribution
    label_indices = np.argmax(np.array(labels), axis=1)
    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']
    print("\nEmotion Distribution:")
    for i in range(NUM_CLASSES):
        count = np.sum(label_indices == i)
        print(f"  {emotion_names[i]}: {count} ({count/len(labels)*100:.1f}%)")

    return valid_cnn_audio_files, np.array(labels), mapped_labels


def feature_quality_check(file_paths, labels):
    """Check feature quality by plotting statistics about a few samples."""
    print("\nFeature Quality Check:")
    num_samples = min(5, len(file_paths))
    
    for i in range(num_samples):
        try:
            features = np.load(file_paths[i])
            label_idx = np.argmax(labels[i])
            emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']
            
            print(f"\nSample {i+1} - {os.path.basename(file_paths[i])} - Class: {emotion_names[label_idx]}")
            print(f"  Shape: {features.shape}")
            print(f"  Mean: {np.mean(features):.4f}")
            print(f"  Std: {np.std(features):.4f}")
            print(f"  Min: {np.min(features):.4f}")
            print(f"  Max: {np.max(features):.4f}")
            print(f"  % of zeros: {np.sum(features == 0) / features.size * 100:.2f}%")
            
            # Count NaN and Inf values
            nan_count = np.isnan(features).sum()
            inf_count = np.isinf(features).sum()
            if nan_count > 0 or inf_count > 0:
                print(f"  WARNING: Contains {nan_count} NaN and {inf_count} Inf values!")
                
        except Exception as e:
            print(f"Error loading file {file_paths[i]}: {e}")


def select_subset_for_overfitting(cnn_audio_files, labels, samples_per_class=SAMPLES_PER_CLASS):
    """Select a balanced subset of samples for overfitting test."""
    class_indices = [np.where(np.argmax(labels, axis=1) == i)[0] for i in range(NUM_CLASSES)]
    
    subset_files = []
    subset_labels = []
    
    for i, indices in enumerate(class_indices):
        np.random.seed(RANDOM_SEED + i)  # Set seed per class for reproducibility
        if len(indices) >= samples_per_class:
            selected_indices = np.random.choice(indices, samples_per_class, replace=False)
        else:
            # If not enough samples, take all and repeat some
            selected_indices = np.random.choice(indices, samples_per_class, replace=True)
            
        for idx in selected_indices:
            subset_files.append(cnn_audio_files[idx])
            subset_labels.append(labels[idx])
    
    print(f"\nSelected {len(subset_files)} samples for overfitting test "
          f"({samples_per_class} per class)")
    
    return subset_files, np.array(subset_labels)


def overfit_test(subset_files, subset_labels, epochs=50):
    """Run the overfitting test on a small subset of data."""
    print("\n=== OVERFITTING TEST ===")
    print("Loading feature data...")
    
    # Load all features to memory
    X_data = []
    for file_path in tqdm(subset_files):
        try:
            X_data.append(np.load(file_path))
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Add zeros as placeholder for failed loads to avoid array length mismatch
            X_data.append(np.zeros((1, 2048)))
    
    # Create a simple model - similar to the model in the original script but simplified
    feature_dim = X_data[0].shape[1]
    
    # Build a super-simple model that just averages over time
    inputs = Input(shape=(None, feature_dim))
    x = GlobalAveragePooling1D()(inputs)  # Average over time dimension
    x = Dense(128, activation='relu')(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Custom training loop to handle variable-length sequences
    print("\nStarting overfitting training...")
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(len(X_data))
        epoch_loss = 0
        epoch_acc = 0
        
        # Train on each sample individually (simple batch of 1)
        for i in tqdm(indices, desc=f"Epoch {epoch+1}/{epochs}"):
            x = np.expand_dims(X_data[i], axis=0)  # Add batch dimension
            y = np.expand_dims(subset_labels[i], axis=0)
            
            # Train on single sample
            metrics = model.train_on_batch(x, y)
            epoch_loss += metrics[0]
            epoch_acc += metrics[1]
        
        # Calculate average metrics
        epoch_loss /= len(indices)
        epoch_acc /= len(indices)
        
        # Evaluate on all samples
        all_preds = []
        for i in range(len(X_data)):
            x = np.expand_dims(X_data[i], axis=0)
            pred = model.predict(x, verbose=0)
            all_preds.append(pred[0])
        
        all_preds = np.array(all_preds)
        val_acc = np.mean(np.argmax(all_preds, axis=1) == np.argmax(subset_labels, axis=1))
        
        print(f"Epoch {epoch+1}: loss={epoch_loss:.4f}, train_acc={epoch_acc:.4f}, val_acc={val_acc:.4f}")
        
        # Early stopping if we reach very high accuracy
        if val_acc > 0.95:
            print(f"Early stopping at epoch {epoch+1} with validation accuracy {val_acc:.4f}")
            break
    
    # Final evaluation
    all_preds = []
    for i in range(len(X_data)):
        x = np.expand_dims(X_data[i], axis=0)
        pred = model.predict(x, verbose=0)
        all_preds.append(pred[0])
    
    all_preds = np.array(all_preds)
    final_acc = np.mean(np.argmax(all_preds, axis=1) == np.argmax(subset_labels, axis=1))
    
    print(f"\nFinal overfitting accuracy: {final_acc:.4f}")
    print("If this is significantly above chance level (~0.167), "
          "then the features and labels are consistent.")
    
    # Print confusion matrix
    y_true = np.argmax(subset_labels, axis=1)
    y_pred = np.argmax(all_preds, axis=1)
    
    print("\nConfusion Matrix:")
    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']
    conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for i in range(len(y_true)):
        conf_matrix[y_true[i], y_pred[i]] += 1
    
    print("Predicted")
    print("      " + " ".join(f"{name[:3]:^7}" for name in emotion_names))
    print("     " + "-" * (7 * NUM_CLASSES + 1))
    
    for i in range(NUM_CLASSES):
        print(f"{emotion_names[i][:3]:^5}|" + " ".join(f"{conf_matrix[i, j]:^7}" for j in range(NUM_CLASSES)))


def logistic_regression_baseline(cnn_audio_files, labels):
    """Run a simple logistic regression on time-averaged features."""
    print("\n=== LOGISTIC REGRESSION BASELINE ===")
    print("Extracting time-averaged features...")
    
    # Extract time-averaged features
    X_avg = []
    for file_path in tqdm(cnn_audio_files):
        try:
            features = np.load(file_path)
            # Average over time dimension
            avg_features = np.mean(features, axis=0)
            X_avg.append(avg_features)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Use zeros if loading fails
            X_avg.append(np.zeros(2048))
    
    X_avg = np.array(X_avg)
    y = np.argmax(labels, axis=1)
    
    # Split into train/test (80/20)
    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(len(X_avg))
    split_idx = int(len(indices) * 0.8)
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    X_train, X_test = X_avg[train_idx], X_avg[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"Training logistic regression on {len(X_train)} samples...")
    
    # Train logistic regression
    clf = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        C=1.0,
        max_iter=1000,
        random_state=RANDOM_SEED
    )
    
    # Check for any NaN or Inf values
    has_nan = np.isnan(X_train).any()
    has_inf = np.isinf(X_train).any()
    
    if has_nan or has_inf:
        print("WARNING: Training data contains NaN or Inf values. Replacing with zeros.")
        X_train = np.nan_to_num(X_train)
        X_test = np.nan_to_num(X_test)
    
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nLogistic regression accuracy: {accuracy:.4f}")
    print("If this is well above chance level (~0.167), then the features contain useful signal.")
    
    # Classification report
    print("\nClassification Report:")
    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']
    report = classification_report(
        y_test, y_pred,
        target_names=emotion_names,
        digits=3
    )
    print(report)


def main():
    """Run the diagnostic tests."""
    print("\n=== EMOTION RECOGNITION CNN FEATURE DIAGNOSTIC ===\n")
    
    # Load data
    cnn_audio_files, labels, mapped_labels = load_data_paths_and_labels_audio_only(
        RAVDESS_CNN_AUDIO_DIR, CREMA_D_CNN_AUDIO_DIR
    )
    
    # Check for empty directories
    if len(cnn_audio_files) == 0:
        print("ERROR: No CNN audio files found. Check if the directories exist and contain .npy files.")
        print(f"RAVDESS_CNN_AUDIO_DIR: {RAVDESS_CNN_AUDIO_DIR}")
        print(f"CREMA_D_CNN_AUDIO_DIR: {CREMA_D_CNN_AUDIO_DIR}")
        sys.exit(1)
    
    # Check a few features to understand their structure
    feature_quality_check(cnn_audio_files, labels)
    
    # Try creating empty/dummy subdirectories to solve path issues
    for directory in [RAVDESS_CNN_AUDIO_DIR, CREMA_D_CNN_AUDIO_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    # Run the tests
    try:
        # Test 1: LogisticRegression baseline
        if len(cnn_audio_files) > 10:
            logistic_regression_baseline(cnn_audio_files, labels)
        else:
            print("\nSkipping logistic regression test due to insufficient data.")
            
        # Test 2: Overfitting on a small subset
        if sum(np.sum(labels, axis=0) >= SAMPLES_PER_CLASS) == NUM_CLASSES:
            subset_files, subset_labels = select_subset_for_overfitting(cnn_audio_files, labels)
            overfit_test(subset_files, subset_labels)
        else:
            print("\nSkipping overfitting test - not enough samples per class.")
    
    except Exception as e:
        print(f"Error during diagnostic tests: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
