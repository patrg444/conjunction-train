#!/usr/bin/env python3
"""
Visualization utility for model architecture and training results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

def visualize_model_architecture(model, output_file="model_architecture.png", show_shapes=True, show_layer_names=True):
    """Visualizes the model architecture and saves it as an image.

    Args:
        model: Keras model to visualize.
        output_file: Path to save the architecture image.
        show_shapes: Whether to display shape information.
        show_layer_names: Whether to display layer names.
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    # Plot the model
    plot_model(model, to_file=output_file, show_shapes=show_shapes, show_layer_names=show_layer_names)
    print(f"Model architecture visualization saved to {output_file}")

def plot_training_history(history, output_file="training_history.png"):
    """Plots the training history (accuracy and loss) and saves it as an image.

    Args:
        history: History object returned by model.fit().
        output_file: Path to save the plot image.
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot training & validation accuracy
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title('Model Accuracy')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend(loc='lower right')
    axes[0].grid(True)
    
    # Plot training & validation loss
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title('Model Loss')
    axes[1].set_ylabel('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(loc='upper right')
    axes[1].grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Training history plot saved to {output_file}")
    plt.close()

def visualize_confusion_matrix(y_true, y_pred, class_names, output_file="confusion_matrix.png"):
    """Visualizes the confusion matrix and saves it as an image.

    Args:
        y_true: True labels (one-hot encoded).
        y_pred: Predicted labels (model outputs).
        class_names: List of class names.
        output_file: Path to save the confusion matrix image.
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    # Convert from one-hot encoding to class indices
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Normalize the confusion matrix by row (i.e., by the number of samples in each true class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create a heatmap
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Normalized Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Confusion matrix visualization saved to {output_file}")
    plt.close()
    
    # Also save the raw confusion matrix data
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_csv_file = output_file.replace('.png', '.csv')
    cm_df.to_csv(cm_csv_file)
    print(f"Confusion matrix data saved to {cm_csv_file}")

def visualize_attention_weights(model, X_sample, layer_name, output_file="attention_weights.png"):
    """Visualizes the attention weights for a sample input.

    Args:
        model: Keras model with an attention layer.
        X_sample: Sample input data [video_input, audio_input].
        layer_name: Name of the attention layer.
        output_file: Path to save the attention weights visualization.
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    try:
        # Get the attention layer
        attention_layer = model.get_layer(layer_name)
        
        # Create a new model that outputs the attention weights
        attention_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=attention_layer.output
        )
        
        # Get attention weights for the sample input
        attention_weights = attention_model.predict(X_sample)
        
        # Visualize the attention weights
        plt.figure(figsize=(10, 6))
        plt.imshow(attention_weights[0], aspect='auto', cmap='viridis')
        plt.colorbar(label='Attention weight')
        plt.xlabel('Feature dimension')
        plt.ylabel('Time step')
        plt.title(f'Attention Weights for Layer: {layer_name}')
        
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"Attention weights visualization saved to {output_file}")
        plt.close()
        
    except Exception as e:
        print(f"Error visualizing attention weights: {str(e)}")

def visualize_feature_maps(model, X_sample, layer_name, output_file="feature_maps.png", max_maps=16):
    """Visualizes the feature maps for a sample input.

    Args:
        model: Keras model.
        X_sample: Sample input data [video_input, audio_input].
        layer_name: Name of the layer to visualize.
        output_file: Path to save the feature maps visualization.
        max_maps: Maximum number of feature maps to display.
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    try:
        # Create a model that outputs the feature maps of the specified layer
        feature_map_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=model.get_layer(layer_name).output
        )
        
        # Get feature maps for the sample input
        feature_maps = feature_map_model.predict(X_sample)
        
        # Determine the number of feature maps to display
        n_feature_maps = min(max_maps, feature_maps.shape[-1])
        
        # Calculate grid dimensions
        grid_size = int(np.ceil(np.sqrt(n_feature_maps)))
        
        # Create figure
        plt.figure(figsize=(15, 15))
        
        # Plot each feature map
        for i in range(n_feature_maps):
            plt.subplot(grid_size, grid_size, i+1)
            plt.imshow(feature_maps[0, :, :, i], aspect='auto', cmap='viridis')
            plt.title(f'Map {i+1}')
            plt.xticks([])
            plt.yticks([])
        
        plt.suptitle(f'Feature Maps for Layer: {layer_name}')
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"Feature maps visualization saved to {output_file}")
        plt.close()
        
    except Exception as e:
        print(f"Error visualizing feature maps: {str(e)}")

def visualize_model_performance(model_path, X_test, y_test, class_names, output_dir="visualizations"):
    """Generates and saves visualizations for model performance.

    Args:
        model_path: Path to the saved model.
        X_test: Test input data [video_input, audio_input].
        y_test: Test labels.
        class_names: List of class names.
        output_dir: Directory to save the visualizations.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    model = load_model(model_path)
    
    # Visualize the model architecture
    visualize_model_architecture(model, output_file=os.path.join(output_dir, "model_architecture.png"))
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Visualize the confusion matrix
    visualize_confusion_matrix(y_test, y_pred, class_names, 
                              output_file=os.path.join(output_dir, "confusion_matrix.png"))
    
    # Print classification report
    y_true_classes = np.argmax(y_test, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    report = classification_report(y_true_classes, y_pred_classes, target_names=class_names)
    print("Classification Report:")
    print(report)
    
    # Save the classification report to a file
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

def main():
    print("Model visualization utility")
    print("This script can be imported and used in other scripts to visualize model architecture and results.")

if __name__ == "__main__":
    main()
