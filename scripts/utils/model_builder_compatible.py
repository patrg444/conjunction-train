#!/usr/bin/env python
"""
Model Builder Compatible with Modern TensorFlow

This script recreates the exact branched model architecture from train_branched_no_leakage.py
but with compatibility fixes for modern TensorFlow versions.
"""

import tensorflow as tf
import os
import logging
import numpy as np
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CustomLSTM(tf.keras.layers.LSTM):
    """Custom LSTM layer that ignores time_major parameter for TF compatibility."""
    
    def __init__(self, *args, **kwargs):
        # Remove time_major if present
        if 'time_major' in kwargs:
            logger.info(f"Removing time_major parameter")
            kwargs.pop('time_major')
        super().__init__(*args, **kwargs)


def create_compatible_model(audio_feature_dim=89, video_feature_dim=512, num_classes=6):
    """
    Create a compatible version of the same branched model architecture.
    
    This exactly matches the architecture from train_branched_no_leakage.py but uses
    custom LSTM layers that ignore the time_major parameter.
    
    Args:
        audio_feature_dim: Number of audio features (default: 89 for eGeMAPSv02)
        video_feature_dim: Number of video features (default: 512 for FaceNet)
        num_classes: Number of emotion classes (default: 6)
        
    Returns:
        A compiled TensorFlow model
    """
    logger.info(f"Creating compatible branched model:")
    logger.info(f"- Audio feature dimension: {audio_feature_dim}")
    logger.info(f"- Video feature dimension: {video_feature_dim}")
    logger.info(f"- Number of classes: {num_classes}")

    # Audio branch with masking
    audio_input = tf.keras.layers.Input(shape=(None, audio_feature_dim), name='audio_input')

    # Add masking layer to handle padding
    audio_masked = tf.keras.layers.Masking(mask_value=0.0)(audio_input)

    # Apply 1D convolutions to extract local patterns
    audio_x = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(audio_masked)
    audio_x = tf.keras.layers.BatchNormalization()(audio_x)
    audio_x = tf.keras.layers.MaxPooling1D(pool_size=2)(audio_x)

    audio_x = tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(audio_x)
    audio_x = tf.keras.layers.BatchNormalization()(audio_x)
    audio_x = tf.keras.layers.MaxPooling1D(pool_size=2)(audio_x)

    # Apply bidirectional LSTM for temporal features - using our compatible version
    audio_x = tf.keras.layers.Bidirectional(CustomLSTM(128, return_sequences=True))(audio_x)
    audio_x = tf.keras.layers.Dropout(0.3)(audio_x)
    audio_x = tf.keras.layers.Bidirectional(CustomLSTM(64, return_sequences=False))(audio_x)
    audio_x = tf.keras.layers.Dense(128, activation='relu')(audio_x)
    audio_x = tf.keras.layers.Dropout(0.4)(audio_x)

    # Video branch with masking
    video_input = tf.keras.layers.Input(shape=(None, video_feature_dim), name='video_input')

    # Add masking layer to handle padding
    video_masked = tf.keras.layers.Masking(mask_value=0.0)(video_input)

    # FaceNet features already have high dimensionality so we'll use LSTM directly
    video_x = tf.keras.layers.Bidirectional(CustomLSTM(256, return_sequences=True))(video_masked)
    video_x = tf.keras.layers.Dropout(0.3)(video_x)
    video_x = tf.keras.layers.Bidirectional(CustomLSTM(128, return_sequences=False))(video_x)
    video_x = tf.keras.layers.Dense(256, activation='relu')(video_x)
    video_x = tf.keras.layers.Dropout(0.4)(video_x)

    # Merge branches with more sophisticated fusion
    merged = tf.keras.layers.Concatenate()([audio_x, video_x])
    merged = tf.keras.layers.Dense(256, activation='relu')(merged)
    merged = tf.keras.layers.BatchNormalization()(merged)
    merged = tf.keras.layers.Dropout(0.5)(merged)
    merged = tf.keras.layers.Dense(128, activation='relu')(merged)
    merged = tf.keras.layers.BatchNormalization()(merged)
    merged = tf.keras.layers.Dropout(0.4)(merged)

    # Output layer
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(merged)

    # Create model
    model = tf.keras.models.Model(inputs=[video_input, audio_input], outputs=output)

    # Compile model with reduced learning rate for better convergence
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def load_compatible_model(model_path, audio_dim=89, video_dim=512, num_classes=6):
    """
    Load a model checkpoint by creating a compatible architecture and loading weights.
    
    Args:
        model_path: Path to the saved model file
        audio_dim: Audio feature dimension
        video_dim: Video feature dimension
        num_classes: Number of emotion classes
        
    Returns:
        A loaded model with weights from the checkpoint
    """
    try:
        logger.info(f"Creating compatible model architecture...")
        model = create_compatible_model(audio_dim, video_dim, num_classes)
        
        logger.info(f"Loading weights from: {model_path}")
        model.load_weights(model_path)
        logger.info(f"Successfully loaded model weights!")
        
        return model
    except Exception as e:
        logger.error(f"Error loading compatible model: {str(e)}")
        return None


def main():
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python model_builder_compatible.py <model_path>")
        return 1
    
    model_path = sys.argv[1]
    
    # Print system info
    logger.info(f"Python version: {sys.version}")
    logger.info(f"TensorFlow version: {tf.__version__}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return 1
    
    # Load the model
    model = load_compatible_model(model_path)
    
    if model is None:
        logger.error("Failed to load model")
        return 1
    
    # Print model summary
    logger.info("Model loaded successfully")
    logger.info("Model Summary:")
    model.summary(print_fn=logger.info)
    
    # Print input/output shapes
    input_shapes = [layer.shape for layer in model.inputs]
    output_shapes = [layer.shape for layer in model.outputs]
    logger.info(f"Model input shapes: {input_shapes}")
    logger.info(f"Model output shapes: {output_shapes}")
    
    # Save the compatible model if needed
    output_dir = os.path.dirname(model_path)
    output_path = os.path.join(output_dir, "compatible_model.h5")
    logger.info(f"Saving compatible model to: {output_path}")
    model.save(output_path)
    logger.info(f"Compatible model saved successfully!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
