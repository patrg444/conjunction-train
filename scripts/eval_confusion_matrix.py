#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate and visualize confusion matrix for FaceNet LSTM model.
This script evaluates the trained model on the validation set and
visualizes the confusion matrix to identify misclassification patterns.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.fixed_video_facenet_generator import FixedVideoFacenetGenerator

# Emotion mapping
emotion_names = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad'}

def evaluate_model(model, val_gen, num_batches=None):
    """Evaluate model on validation generator."""
    print("Evaluating model on validation set...")
    
    all_true_labels = []
    all_predictions = []
    
    if num_batches is None:
        num_batches = len(val_gen)
    
    # Process all batches (or specified number)
    for i in range(min(num_batches, len(val_gen))):
        features, labels = next(iter(val_gen))
        
        # Get model predictions
        predictions = model.predict(features, verbose=0)
        pred_labels = np.argmax(predictions, axis=1)
        
        # Append to overall results
        all_true_labels.extend(labels)
        all_predictions.extend(pred_labels)
        
        # Print progress
        if i % 10 == 0:
            print(f"Processed {i+1}/{min(num_batches, len(val_gen))} batches")
    
    all_true_labels = np.array(all_true_labels)
    all_predictions = np.array(all_predictions)
    
    # Calculate accuracy
    accuracy = np.mean(all_true_labels == all_predictions)
    print(f"Overall accuracy: {accuracy:.4f}")
    
    return all_true_labels, all_predictions

def generate_confusion_matrix(true_labels, predictions):
    """Generate and visualize confusion matrix."""
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure and plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_names.values(),
                yticklabels=emotion_names.values(), ax=ax1)
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    ax1.set_title('Confusion Matrix (Counts)')
    
    # Plot percentages
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=emotion_names.values(),
                yticklabels=emotion_names.values(), ax=ax2)
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    ax2.set_title('Confusion Matrix (Normalized)')
    
    # Save the figure
    os.makedirs("model_evaluation", exist_ok=True)
    plt.tight_layout()
    plt.savefig("model_evaluation/confusion_matrix.png")
    print("Confusion matrix saved to model_evaluation/confusion_matrix.png")
    
    # Print classification report
    print("\nClassification Report:")
    target_names = [emotion_names[i] for i in range(len(emotion_names))]
    report = classification_report(true_labels, predictions, target_names=target_names)
    print(report)
    
    # Save report to file
    with open("model_evaluation/classification_report.txt", "w") as f:
        f.write(report)
    
    return cm, cm_normalized

def analyze_misclassifications(true_labels, predictions, threshold=0.15):
    """Analyze common misclassification patterns."""
    print("\n== Common Misclassification Patterns ==")
    
    misclass_counts = {}
    for true_label, pred_label in zip(true_labels, predictions):
        if true_label != pred_label:
            pair = (emotion_names[true_label], emotion_names[pred_label])
            misclass_counts[pair] = misclass_counts.get(pair, 0) + 1
    
    # Sort by frequency
    sorted_misclass = sorted(misclass_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate total number of misclassifications
    total_misclass = sum(misclass_counts.values())
    
    # Print top misclassifications above threshold
    print(f"Top misclassification patterns (> {threshold*100:.1f}% of all errors):")
    for (true_label, pred_label), count in sorted_misclass:
        percentage = count / total_misclass
        if percentage >= threshold:
            print(f"  {true_label} → {pred_label}: {count} instances ({percentage:.1%} of all errors)")
    
    # Calculate per-class error rates
    class_counts = {}
    for label in true_labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    print("\nPer-class error rates:")
    for class_idx in range(len(emotion_names)):
        if class_idx in class_counts:
            class_errors = sum(1 for true, pred in zip(true_labels, predictions) 
                              if true == class_idx and pred != class_idx)
            error_rate = class_errors / class_counts[class_idx]
            print(f"  {emotion_names[class_idx]}: {error_rate:.2%} error rate")

def plot_most_confused_emotions(cm_normalized):
    """Plot the most confused emotion pairs."""
    # Create a matrix without diagonal (correct predictions)
    cm_without_diag = cm_normalized.copy()
    np.fill_diagonal(cm_without_diag, 0)
    
    # Find top 5 confusion pairs
    confused_pairs = []
    for i in range(len(emotion_names)):
        for j in range(len(emotion_names)):
            if i != j:
                confused_pairs.append(((i, j), cm_without_diag[i, j]))
    
    top_confused = sorted(confused_pairs, key=lambda x: x[1], reverse=True)[:5]
    
    # Create bar chart
    plt.figure(figsize=(12, 6))
    pair_labels = [f"{emotion_names[true]} → {emotion_names[pred]}" for (true, pred), _ in top_confused]
    confusion_values = [value for _, value in top_confused]
    
    bars = plt.bar(pair_labels, confusion_values, color='orange')
    
    # Add percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.ylabel('Confusion Rate')
    plt.title('Top 5 Most Confused Emotion Pairs')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("model_evaluation/top_confusion_pairs.png")
    print("Top confusion pairs chart saved to model_evaluation/top_confusion_pairs.png")

def main():
    """Main function to evaluate model and generate confusion matrix."""
    parser = argparse.ArgumentParser(description="Evaluate model and generate confusion matrix")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to trained model")
    parser.add_argument("--ravdess_dir", type=str, default="./ravdess_features_facenet", 
                        help="Directory with RAVDESS features")
    parser.add_argument("--cremad_dir", type=str, default="./crema_d_features_facenet", 
                        help="Directory with CREMA-D features")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for evaluation")
    parser.add_argument("--num_batches", type=int, default=None, 
                        help="Number of batches to evaluate (default: all)")
    args = parser.parse_args()
    
    print("=== FaceNet LSTM Model Evaluation ===")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    try:
        model = tf.keras.models.load_model(args.model_path)
        model.summary()
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Initialize validation generator
    print("\nInitializing validation generator...")
    val_gen = FixedVideoFacenetGenerator(
        ravdess_dir=args.ravdess_dir,
        cremad_dir=args.cremad_dir,
        batch_size=args.batch_size,
        is_training=False,
        normalize_features=True,
        mask_zero_paddings=True
    )
    
    # Evaluate model
    true_labels, predictions = evaluate_model(model, val_gen, args.num_batches)
    
    # Generate confusion matrix
    cm, cm_normalized = generate_confusion_matrix(true_labels, predictions)
    
    # Analyze misclassifications
    analyze_misclassifications(true_labels, predictions)
    
    # Plot most confused emotions
    plot_most_confused_emotions(cm_normalized)
    
    print("\n=== Evaluation Complete ===")

if __name__ == "__main__":
    main()
