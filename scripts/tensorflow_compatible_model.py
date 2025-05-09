"""
TensorFlow Compatible Model for Emotion Recognition

This module provides a compatible wrapper for loading and using models
trained with various TensorFlow versions. It ensures compatibility with 
TensorFlow 2.x by recreating the model architecture and loading weights.
"""

import os
import numpy as np
import tensorflow as tf

class EmotionRecognitionModel:
    """
    A compatible wrapper for emotion recognition models
    that handles TensorFlow version differences.
    """
    
    def __init__(self, model_path):
        """
        Initialize the model by loading weights from file.
        
        Args:
            model_path: Path to the saved model or weights file (.h5)
        """
        self.model_path = model_path
        self.model = self.build_model()
        
    def build_model(self):
        """
        Build the emotion recognition model with the same architecture
        as the original, but compatible with TensorFlow 2.x.
        
        The model has a dual-stream architecture with separate
        branches for audio and video processing.
        """
        # Configure TensorFlow 
        self._configure_tensorflow()
        
        # Define input shapes
        audio_input = tf.keras.layers.Input(shape=(None, 89), name='audio_input')  # 89 openSMILE features (from training model)
        video_input = tf.keras.layers.Input(shape=(None, 512), name='video_input') # 512 FaceNet features
        
        # Audio branch
        audio_masked = tf.keras.layers.Masking(mask_value=0.0)(audio_input)
        
        # 1D convolutions for local pattern extraction
        audio_x = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(audio_masked)
        audio_x = tf.keras.layers.BatchNormalization()(audio_x)
        audio_x = tf.keras.layers.MaxPooling1D(pool_size=2, padding='same')(audio_x)  # Added padding='same' for compatibility
        
        audio_x = tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(audio_x)
        audio_x = tf.keras.layers.BatchNormalization()(audio_x)
        audio_x = tf.keras.layers.MaxPooling1D(pool_size=2, padding='same')(audio_x)  # Added padding='same' for compatibility
        
        # Bidirectional LSTM layers for temporal features
        audio_x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True))(audio_x)
        audio_x = tf.keras.layers.Dropout(0.3)(audio_x)
        audio_x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64))(audio_x)
        audio_x = tf.keras.layers.Dense(128, activation='relu')(audio_x)
        audio_x = tf.keras.layers.Dropout(0.4)(audio_x)
        
        # Video branch
        video_masked = tf.keras.layers.Masking(mask_value=0.0)(video_input)
        
        # FaceNet features already have high dimensionality, use LSTM directly
        video_x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(256, return_sequences=True))(video_masked)
        video_x = tf.keras.layers.Dropout(0.3)(video_x)
        video_x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128))(video_x)
        video_x = tf.keras.layers.Dense(256, activation='relu')(video_x)
        video_x = tf.keras.layers.Dropout(0.4)(video_x)
        
        # Merge branches
        merged = tf.keras.layers.concatenate([audio_x, video_x])
        
        # Fully connected layers for fusion
        merged = tf.keras.layers.Dense(256, activation='relu')(merged)
        merged = tf.keras.layers.BatchNormalization()(merged)
        merged = tf.keras.layers.Dropout(0.5)(merged)
        merged = tf.keras.layers.Dense(128, activation='relu')(merged)
        merged = tf.keras.layers.BatchNormalization()(merged)
        merged = tf.keras.layers.Dropout(0.4)(merged)
        
        # Output layer (6 emotion classes)
        outputs = tf.keras.layers.Dense(6, activation='softmax')(merged)
        
        # Create model with inputs in the same order as train_branched_no_leakage.py
        model = tf.keras.models.Model(inputs=[video_input, audio_input], outputs=outputs)
        
        # Load weights from saved model
        model.load_weights(self.model_path)
        
        return model
        
    def _configure_tensorflow(self):
        """Configure TensorFlow for compatibility."""
        # Allow memory growth to avoid allocating all GPU memory at once
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"GPU memory config error: {e}")
    
    def predict(self, video_features, audio_features):
        """
        Predict emotion from video and audio features.
        
        Args:
            video_features: Video features array [batch_size, seq_len, 512]
            audio_features: Audio features array [batch_size, seq_len, 88]
            
        Returns:
            Array of emotion probabilities [batch_size, 6]
        """
        # Ensure inputs have the right dimensions
        if len(video_features.shape) < 3:
            video_features = np.expand_dims(video_features, axis=0)
        if len(audio_features.shape) < 3:
            audio_features = np.expand_dims(audio_features, axis=0)
            
        # Make prediction - ensuring inputs are in the same order as in training
        prediction = self.model.predict([video_features, audio_features], verbose=0)
        
        return prediction
        
    def get_emotion_label(self, prediction):
        """
        Convert prediction array to emotion label.
        
        Args:
            prediction: Prediction array from model
            
        Returns:
            String with emotion label
        """
        # IMPORTANT: Order must match training data: ANG=0, DIS=1, FEA=2, HAP=3, NEU=4, SAD=5
        emotions = ["anger", "disgust", "fear", "happiness", "neutral", "sadness"]
        emotion_idx = np.argmax(prediction[0])
        return emotions[emotion_idx]

# For testing the model
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python tensorflow_compatible_model.py <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    # Create model
    model = EmotionRecognitionModel(model_path)
    
    # Create dummy inputs for testing
    batch_size = 1
    seq_len = 10
    
    video_features = np.random.random((batch_size, seq_len, 512))  # FaceNet features
    audio_features = np.random.random((batch_size, seq_len, 89))   # OpenSMILE features (89 dimensions from training)
    
    # Test prediction
    prediction = model.predict(video_features, audio_features)
    emotion = model.get_emotion_label(prediction)
    
    print(f"Model loaded successfully from {model_path}")
    print(f"Prediction shape: {prediction.shape}")
    print(f"Predicted emotion: {emotion}")
    print(f"Emotion probabilities: {prediction[0]}")
