#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyzes the quality and separability of precomputed Facenet features for emotion recognition.
Performs:
1. Loads features and labels.
2. Calculates mean feature vector per clip.
3. Trains and evaluates a simple Logistic Regression baseline.
4. Performs PCA and t-SNE dimensionality reduction.
5. Saves plots of reduced features colored by emotion.
"""

import os
import sys
import numpy as np
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import time

# --- Configuration ---
NUM_CLASSES = 6
RANDOM_SEED = 42
TEST_SPLIT_RATIO = 0.2 # Use a standard train/test split for baseline evaluation
# Define feature directories (ABSOLUTE PATHS on EC2)
RAVDESS_FACENET_DIR = "/home/ubuntu/emotion-recognition/ravdess_features_facenet"
CREMA_D_FACENET_DIR = "/home/ubuntu/emotion-recognition/crema_d_features_facenet"
# Output directory for plots
OUTPUT_DIR = "feature_analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Data Loading (Copied and adapted from training script) ---
def load_data_paths_and_labels_video_only(facenet_dir_ravdess, facenet_dir_cremad):
    """Finds precomputed Facenet feature files (.npz) and extracts labels."""
    facenet_files = glob.glob(os.path.join(facenet_dir_ravdess, "Actor_*", "*.npz")) + \
                    glob.glob(os.path.join(facenet_dir_cremad, "*.npz"))

    labels = []
    file_paths = [] # Store paths corresponding to labels
    # Emotion map should match the one used in training
    emotion_map = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}
    ravdess_emotion_map = {'01': 'NEU', '02': 'NEU', '03': 'HAP', '04': 'SAD', '05': 'ANG', '06': 'FEA', '07': 'DIS', '08': 'FEA'}

    print(f"Found {len(facenet_files)} potential Facenet files. Extracting labels and checking validity...")
    skipped_count = 0

    for facenet_file in tqdm(facenet_files, desc="Loading labels"):
        base_name = os.path.splitext(os.path.basename(facenet_file))[0]
        if base_name.endswith('_facenet_features'):
            base_name = base_name[:-len('_facenet_features')]

        label_idx = -1 # Use -1 to indicate no valid label found yet

        try:
            if "Actor_" in facenet_file: # RAVDESS
                parts = base_name.split('-')
                if len(parts) >= 3:
                    emotion_code = ravdess_emotion_map.get(parts[2], None)
                    if emotion_code in emotion_map:
                        label_idx = emotion_map[emotion_code]
            else: # CREMA-D
                parts = base_name.split('_')
                if len(parts) >= 3:
                    emotion_code = parts[2]
                    if emotion_code in emotion_map:
                        label_idx = emotion_map[emotion_code]
        except Exception as e:
            print(f"Label parsing error for {facenet_file}: {e}")
            label_idx = -1

        if label_idx != -1:
            # Check file existence and basic validity before adding
            if os.path.exists(facenet_file) and os.path.getsize(facenet_file) > 0:
                 # Further check: can we load 'video_features'?
                 try:
                      with np.load(facenet_file, allow_pickle=True) as data:
                           if 'video_features' in data and data['video_features'].shape[0] > 0:
                                file_paths.append(facenet_file)
                                labels.append(label_idx) # Store index directly
                           else:
                               # print(f"Warning: Skipping {facenet_file} - 'video_features' key missing or empty.")
                                skipped_count += 1
                 except Exception as load_e:
                      # print(f"Warning: Skipping {facenet_file} - Error loading npz: {load_e}")
                      skipped_count += 1
            else:
                # print(f"Warning: Skipping {facenet_file} - File does not exist or is empty.")
                skipped_count += 1
        else:
            skipped_count += 1

    print(f"Found {len(file_paths)} Facenet files with valid labels and features.")
    print(f"Skipped {skipped_count} files.")

    if not file_paths:
        raise FileNotFoundError("No valid Facenet files found.")

    # Convert labels to numpy array of integers
    return file_paths, np.array(labels, dtype=int)

# --- Feature Extraction (Mean Pooling) ---
def extract_mean_features(file_paths):
    """Loads Facenet features and computes the mean vector for each file."""
    mean_features = []
    print("Extracting mean features from NPZ files...")
    for f_path in tqdm(file_paths, desc="Calculating mean features"):
        try:
            with np.load(f_path, allow_pickle=True) as data:
                if 'video_features' in data:
                    features = data['video_features'].astype(np.float32)
                    if features.shape[0] > 0:
                        mean_features.append(np.mean(features, axis=0))
                    else:
                        # Handle empty features - append zeros or skip? Append zeros for now.
                        print(f"Warning: Empty features in {f_path}. Appending zeros.")
                        # Need to know the feature dimension. Assume 512 if first file worked.
                        feature_dim = mean_features[0].shape[0] if mean_features else 512
                        mean_features.append(np.zeros(feature_dim, dtype=np.float32))
                else:
                     print(f"Warning: 'video_features' key not found in {f_path}. Appending zeros.")
                     feature_dim = mean_features[0].shape[0] if mean_features else 512
                     mean_features.append(np.zeros(feature_dim, dtype=np.float32))
        except Exception as e:
            print(f"Error loading {f_path}: {e}. Appending zeros.")
            feature_dim = mean_features[0].shape[0] if mean_features else 512
            mean_features.append(np.zeros(feature_dim, dtype=np.float32))

    return np.array(mean_features)

# --- Baseline Model ---
def train_evaluate_baseline(features, labels):
    """Trains and evaluates a Logistic Regression model."""
    print("\n--- Training Baseline Logistic Regression Model ---")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=TEST_SPLIT_RATIO, random_state=RANDOM_SEED, stratify=labels
    )

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Logistic Regression
    print("Training Logistic Regression...")
    start_time = time.time()
    # Increased max_iter, consider 'saga' solver for large datasets if needed
    model = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000, C=0.1, solver='liblinear')
    model.fit(X_train_scaled, y_train)
    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")

    # Evaluate
    print("\nEvaluating baseline model...")
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Baseline Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD'])) # Use emotion names

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD'], yticklabels=['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD'])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Baseline Confusion Matrix (Mean Facenet Features)")
    cm_path = os.path.join(OUTPUT_DIR, "baseline_confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    plt.close()

    return accuracy

# --- Dimensionality Reduction and Visualization ---
def visualize_features(features, labels, method='PCA'):
    """Performs dimensionality reduction and saves a 2D scatter plot."""
    print(f"\n--- Performing {method} Dimensionality Reduction ---")
    start_time = time.time()

    # Scale features before reduction
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    if method == 'PCA':
        reducer = PCA(n_components=2, random_state=RANDOM_SEED)
    elif method == 'TSNE':
        # t-SNE can be slow, consider reducing perplexity or iterations for large datasets
        reducer = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=30, n_iter=300, init='pca', learning_rate='auto')
    else:
        raise ValueError("Unsupported reduction method")

    features_reduced = reducer.fit_transform(features_scaled)
    end_time = time.time()
    print(f"{method} finished in {end_time - start_time:.2f} seconds.")

    # Plot
    plt.figure(figsize=(12, 10))
    emotion_names = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']
    colors = sns.color_palette("husl", NUM_CLASSES)
    for i in range(NUM_CLASSES):
        idx = np.where(labels == i)[0]
        plt.scatter(features_reduced[idx, 0], features_reduced[idx, 1], c=[colors[i]], label=emotion_names[i], alpha=0.6, s=10)

    plt.title(f'{method} of Mean Facenet Features by Emotion')
    plt.xlabel(f'{method} Component 1')
    plt.ylabel(f'{method} Component 2')
    plt.legend(title="Emotion", markerscale=2)
    plot_path = os.path.join(OUTPUT_DIR, f"facenet_features_{method.lower()}.png")
    plt.savefig(plot_path)
    print(f"{method} plot saved to {plot_path}")
    plt.close()


# --- Main Execution ---
if __name__ == '__main__':
    print("Starting Facenet Feature Quality Analysis...")
    try:
        # 1. Load data
        all_files, all_labels_int = load_data_paths_and_labels_video_only(
            RAVDESS_FACENET_DIR, CREMA_D_FACENET_DIR
        )

        # 2. Extract mean features
        mean_features = extract_mean_features(all_files)
        print(f"Extracted mean features shape: {mean_features.shape}") # Should be (num_samples, 512)

        # 3. Train and evaluate baseline
        baseline_accuracy = train_evaluate_baseline(mean_features, all_labels_int)

        # 4. Visualize features (PCA)
        visualize_features(mean_features, all_labels_int, method='PCA')

        # 5. Visualize features (t-SNE) - Optional, can be slow
        # Consider running t-SNE on a subset if it takes too long
        subset_size_tsne = 5000 # Adjust as needed
        if len(mean_features) > subset_size_tsne:
             print(f"\nRunning t-SNE on a subset of {subset_size_tsne} samples for efficiency.")
             indices_subset = np.random.choice(len(mean_features), subset_size_tsne, replace=False)
             visualize_features(mean_features[indices_subset], all_labels_int[indices_subset], method='TSNE')
        else:
             visualize_features(mean_features, all_labels_int, method='TSNE')


        print("\nAnalysis Complete.")
        print(f"Baseline Logistic Regression Accuracy: {baseline_accuracy:.4f}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure Facenet feature extraction has been run and the paths are correct.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
