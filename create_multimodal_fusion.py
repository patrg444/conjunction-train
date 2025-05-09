#!/usr/bin/env python3
"""
Create a fusion model combining SlowFast video and Audio LSTM models.

This script:
1. Loads pretrained SlowFast video-only and Audio-only models
2. Implements both late fusion (ensemble) and early fusion (joint) approaches
3. Provides inference methods for combined model prediction

Usage:
  python create_multimodal_fusion.py --video_model models/slowfast_emotion_video_only_92.9.pt 
                                    --audio_model models/audio_only_model_*.h5
                                    --output_dir models/fusion
                                    --fusion_type late
"""

import os
import sys
import argparse
import numpy as np
import torch
import tensorflow as tf
from tensorflow import keras
import json

# Configure TensorFlow to not take all GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Emotion labels (shared between both models)
EMOTION_LABELS = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad']

class LateFusionModel:
    """
    Ensemble model that combines predictions from video and audio models.
    
    This implements a "late fusion" approach where each model makes independent
    predictions, and the results are combined through weighted averaging.
    """
    
    def __init__(self, video_model_path, audio_model_path, video_weight=0.6, audio_weight=0.4):
        """
        Initialize the late fusion model.
        
        Args:
            video_model_path: Path to the PyTorch SlowFast model weights
            audio_model_path: Path to the TensorFlow/Keras audio model
            video_weight: Weight for video model predictions (0.0 to 1.0)
            audio_weight: Weight for audio model predictions (0.0 to 1.0)
        """
        self.video_weight = video_weight
        self.audio_weight = audio_weight
        
        # Normalize weights if they don't sum to 1
        total_weight = video_weight + audio_weight
        if total_weight != 1.0:
            self.video_weight /= total_weight
            self.audio_weight /= total_weight
            
        print(f"Fusion weights: Video={self.video_weight:.2f}, Audio={self.audio_weight:.2f}")
        
        # Load video model
        print(f"Loading video model from {video_model_path}")
        self.video_model = self._load_video_model(video_model_path)
        
        # Load audio model
        print(f"Loading audio model from {audio_model_path}")
        self.audio_model = self._load_audio_model(audio_model_path)
        
        # Load normalization parameters for audio
        audio_dir = os.path.dirname(audio_model_path)
        try:
            self.audio_mean = np.load(os.path.join(audio_dir, 'audio_mean.npy'))
            self.audio_std = np.load(os.path.join(audio_dir, 'audio_std.npy'))
            print("Loaded audio normalization parameters")
        except:
            print("Warning: Could not load audio normalization parameters")
            self.audio_mean = None
            self.audio_std = None
    
    def _load_video_model(self, model_path):
        """Load the PyTorch SlowFast model."""
        try:
            # Import here to avoid dependency issues if only using audio model
            from scripts.train_slowfast_emotion import EmotionClassifier
            
            # Initialize model architecture
            model = EmotionClassifier(
                num_classes=len(EMOTION_LABELS),
                pretrained=False
            )
            
            # Load weights
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()
            
            # Use GPU if available
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            return model
        except Exception as e:
            print(f"Error loading video model: {e}")
            return None
    
    def _load_audio_model(self, model_path):
        """Load the TensorFlow/Keras audio model."""
        try:
            # Load the keras model
            model = keras.models.load_model(model_path)
            return model
        except Exception as e:
            print(f"Error loading audio model: {e}")
            return None
    
    def _preprocess_audio(self, audio_features):
        """Preprocess audio features."""
        if self.audio_mean is not None and self.audio_std is not None:
            return (audio_features - self.audio_mean) / self.audio_std
        return audio_features
    
    def predict(self, video_frames, audio_features):
        """
        Make a prediction using both models and combine results.
        
        Args:
            video_frames: Tensor of shape [T, C, H, W] for video
            audio_features: Numpy array of shape [T_audio, audio_dim] for audio
        
        Returns:
            predicted_label: String label of the predicted emotion
            confidence: Confidence score (0.0 to 1.0)
            all_probs: Dictionary mapping each emotion to its probability
        """
        # Process video if available
        video_probs = None
        if self.video_model is not None and video_frames is not None:
            try:
                # Add batch dimension
                if len(video_frames.shape) == 4:  # [T, C, H, W]
                    video_frames = video_frames.unsqueeze(0)  # [1, T, C, H, W]
                
                # Move to device
                device = next(self.video_model.parameters()).device
                video_frames = video_frames.to(device)
                
                # Get predictions
                with torch.no_grad():
                    video_logits = self.video_model(video_frames)
                    video_probs = torch.softmax(video_logits, dim=1).cpu().numpy()[0]
            except Exception as e:
                print(f"Error processing video: {e}")
        
        # Process audio if available
        audio_probs = None
        if self.audio_model is not None and audio_features is not None:
            try:
                # Preprocess
                audio_features = self._preprocess_audio(audio_features)
                
                # Add batch dimension
                if len(audio_features.shape) == 2:  # [T, audio_dim]
                    audio_features = np.expand_dims(audio_features, 0)  # [1, T, audio_dim]
                
                # Get predictions
                audio_logits = self.audio_model.predict(audio_features, verbose=0)
                audio_probs = audio_logits[0]
            except Exception as e:
                print(f"Error processing audio: {e}")
        
        # Combine predictions
        combined_probs = None
        if video_probs is not None and audio_probs is not None:
            # Weighted average
            combined_probs = self.video_weight * video_probs + self.audio_weight * audio_probs
        elif video_probs is not None:
            combined_probs = video_probs
        elif audio_probs is not None:
            combined_probs = audio_probs
        else:
            raise ValueError("Both models failed to produce predictions")
        
        # Get predicted class and confidence
        predicted_idx = np.argmax(combined_probs)
        predicted_label = EMOTION_LABELS[predicted_idx]
        confidence = combined_probs[predicted_idx]
        
        # Create dictionary of all probabilities
        all_probs = {label: float(combined_probs[i]) for i, label in enumerate(EMOTION_LABELS)}
        
        return predicted_label, float(confidence), all_probs


class EarlyFusionModel:
    """
    Joint model that combines features from video and audio models.
    
    This implements an "early fusion" approach where features from both models
    are extracted and concatenated before the final classification.
    """
    
    def __init__(self, video_model_path, audio_model_path, joint_model_path=None):
        """
        Initialize the early fusion model.
        
        Args:
            video_model_path: Path to the PyTorch SlowFast model weights
            audio_model_path: Path to the TensorFlow/Keras audio model
            joint_model_path: Path to the joint model (if already trained)
        """
        # Load base models
        print(f"Loading video model from {video_model_path}")
        self.video_model = self._load_video_model(video_model_path)
        
        print(f"Loading audio model from {audio_model_path}")
        self.audio_model = self._load_audio_model(audio_model_path)
        
        # Load joint model if provided
        self.joint_model = None
        if joint_model_path and os.path.exists(joint_model_path):
            print(f"Loading joint model from {joint_model_path}")
            try:
                self.joint_model = keras.models.load_model(joint_model_path)
            except Exception as e:
                print(f"Error loading joint model: {e}")
        
        # Create joint model if not loaded
        if self.joint_model is None:
            print("Creating new joint model")
            self.joint_model = self._create_joint_model()
        
        # Load normalization parameters for audio
        audio_dir = os.path.dirname(audio_model_path)
        try:
            self.audio_mean = np.load(os.path.join(audio_dir, 'audio_mean.npy'))
            self.audio_std = np.load(os.path.join(audio_dir, 'audio_std.npy'))
            print("Loaded audio normalization parameters")
        except:
            print("Warning: Could not load audio normalization parameters")
            self.audio_mean = None
            self.audio_std = None
    
    def _load_video_model(self, model_path):
        """Load the PyTorch SlowFast model."""
        try:
            # Import here to avoid dependency issues if only using audio model
            from scripts.train_slowfast_emotion import EmotionClassifier
            
            # Initialize model architecture
            model = EmotionClassifier(
                num_classes=len(EMOTION_LABELS),
                pretrained=False
            )
            
            # Load weights
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()
            
            # Use GPU if available
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            return model
        except Exception as e:
            print(f"Error loading video model: {e}")
            return None
    
    def _load_audio_model(self, model_path):
        """Load the TensorFlow/Keras audio model."""
        try:
            # Load the keras model
            model = keras.models.load_model(model_path)
            return model
        except Exception as e:
            print(f"Error loading audio model: {e}")
            return None
    
    def _create_joint_model(self):
        """Create a new joint model combining features from both models."""
        # Determine feature dimensions
        video_dim = 512  # Typical for SlowFast/R3D embedding
        audio_dim = 128  # Typical for audio LSTM output
        
        # Create keras model
        video_input = keras.layers.Input(shape=(video_dim,), name='video_features')
        audio_input = keras.layers.Input(shape=(audio_dim,), name='audio_features')
        
        # Concatenate features
        concat = keras.layers.Concatenate()([video_input, audio_input])
        
        # Dense layers
        x = keras.layers.Dense(256, activation='relu')(concat)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.5)(x)
        
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.3)(x)
        
        # Output layer
        output = keras.layers.Dense(len(EMOTION_LABELS), activation='softmax')(x)
        
        # Create model
        model = keras.Model(inputs=[video_input, audio_input], outputs=output)
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def extract_video_features(self, video_frames):
        """Extract features from video frames using the video model."""
        if self.video_model is None or video_frames is None:
            return None
        
        try:
            # Add batch dimension
            if len(video_frames.shape) == 4:  # [T, C, H, W]
                video_frames = video_frames.unsqueeze(0)  # [1, T, C, H, W]
            
            # Move to device
            device = next(self.video_model.parameters()).device
            video_frames = video_frames.to(device)
            
            # Extract features (bypassing the final classification layer)
            with torch.no_grad():
                # For the SlowFast model, we need to access the embedder
                features = self.video_model.video_embedder(video_frames)
                return features.cpu().numpy()
        except Exception as e:
            print(f"Error extracting video features: {e}")
            return None
    
    def extract_audio_features(self, audio_features):
        """Extract features from audio using the audio model."""
        if self.audio_model is None or audio_features is None:
            return None
        
        try:
            # Preprocess
            if self.audio_mean is not None and self.audio_std is not None:
                audio_features = (audio_features - self.audio_mean) / self.audio_std
            
            # Add batch dimension
            if len(audio_features.shape) == 2:  # [T, audio_dim]
                audio_features = np.expand_dims(audio_features, 0)  # [1, T, audio_dim]
            
            # Create a modified model that outputs the features before the final layer
            feature_model = keras.Model(
                inputs=self.audio_model.input,
                outputs=self.audio_model.layers[-2].output
            )
            
            # Extract features
            features = feature_model.predict(audio_features, verbose=0)
            return features
        except Exception as e:
            print(f"Error extracting audio features: {e}")
            return None
    
    def predict(self, video_frames, audio_features):
        """
        Make a prediction using the joint model.
        
        Args:
            video_frames: Tensor of shape [T, C, H, W] for video
            audio_features: Numpy array of shape [T_audio, audio_dim] for audio
        
        Returns:
            predicted_label: String label of the predicted emotion
            confidence: Confidence score (0.0 to 1.0)
            all_probs: Dictionary mapping each emotion to its probability
        """
        # Extract features
        video_feats = self.extract_video_features(video_frames)
        audio_feats = self.extract_audio_features(audio_features)
        
        if video_feats is None or audio_feats is None:
            raise ValueError("Failed to extract features from both modalities")
        
        # Make prediction
        probs = self.joint_model.predict(
            {'video_features': video_feats, 'audio_features': audio_feats},
            verbose=0
        )[0]
        
        # Get predicted class and confidence
        predicted_idx = np.argmax(probs)
        predicted_label = EMOTION_LABELS[predicted_idx]
        confidence = probs[predicted_idx]
        
        # Create dictionary of all probabilities
        all_probs = {label: float(probs[i]) for i, label in enumerate(EMOTION_LABELS)}
        
        return predicted_label, float(confidence), all_probs
    
    def train(self, train_data, val_data, epochs=10, batch_size=32, save_path=None):
        """
        Train the joint model on the provided data.
        
        Args:
            train_data: Dictionary with keys 'video_features', 'audio_features', 'labels'
            val_data: Dictionary with keys 'video_features', 'audio_features', 'labels'
            epochs: Number of training epochs
            batch_size: Batch size
            save_path: Path to save the trained model
        
        Returns:
            history: Training history
        """
        # Train the model
        history = self.joint_model.fit(
            {'video_features': train_data['video_features'], 'audio_features': train_data['audio_features']},
            train_data['labels'],
            validation_data=(
                {'video_features': val_data['video_features'], 'audio_features': val_data['audio_features']},
                val_data['labels']
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
            ]
        )
        
        # Save the model if a path is provided
        if save_path:
            self.joint_model.save(save_path)
            print(f"Model saved to {save_path}")
        
        return history


def save_fusion_model(model, save_dir):
    """
    Save the fusion model to disk.
    
    Args:
        model: The fusion model (late or early)
        save_dir: Directory to save the model
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # For late fusion, just save the weights
    if isinstance(model, LateFusionModel):
        config = {
            'type': 'late',
            'video_weight': model.video_weight,
            'audio_weight': model.audio_weight
        }
        with open(os.path.join(save_dir, 'fusion_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    # For early fusion, save the joint model
    elif isinstance(model, EarlyFusionModel):
        if model.joint_model:
            model.joint_model.save(os.path.join(save_dir, 'joint_model'))
            
            config = {
                'type': 'early'
            }
            with open(os.path.join(save_dir, 'fusion_config.json'), 'w') as f:
                json.dump(config, f, indent=2)


def load_fusion_model(video_model_path, audio_model_path, fusion_dir):
    """
    Load a fusion model from disk.
    
    Args:
        video_model_path: Path to the video model weights
        audio_model_path: Path to the audio model
        fusion_dir: Directory containing the fusion model
    
    Returns:
        The loaded fusion model
    """
    # Load fusion config
    config_path = os.path.join(fusion_dir, 'fusion_config.json')
    if not os.path.exists(config_path):
        raise ValueError(f"Fusion config not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create the appropriate fusion model
    if config['type'] == 'late':
        return LateFusionModel(
            video_model_path=video_model_path,
            audio_model_path=audio_model_path,
            video_weight=config.get('video_weight', 0.6),
            audio_weight=config.get('audio_weight', 0.4)
        )
    elif config['type'] == 'early':
        joint_model_path = os.path.join(fusion_dir, 'joint_model')
        return EarlyFusionModel(
            video_model_path=video_model_path,
            audio_model_path=audio_model_path,
            joint_model_path=joint_model_path
        )
    else:
        raise ValueError(f"Unknown fusion type: {config['type']}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Create a fusion model for emotion recognition')
    parser.add_argument('--video_model', type=str, required=True, help='Path to SlowFast video model weights')
    parser.add_argument('--audio_model', type=str, required=True, help='Path to audio model')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save fusion model')
    parser.add_argument('--fusion_type', type=str, default='late', choices=['late', 'early'],
                      help='Type of fusion: "late" (ensemble) or "early" (joint)')
    parser.add_argument('--video_weight', type=float, default=0.6, help='Weight for video predictions (late fusion)')
    parser.add_argument('--audio_weight', type=float, default=0.4, help='Weight for audio predictions (late fusion)')
    args = parser.parse_args()
    
    # Create the fusion model
    if args.fusion_type == 'late':
        model = LateFusionModel(
            video_model_path=args.video_model,
            audio_model_path=args.audio_model,
            video_weight=args.video_weight,
            audio_weight=args.audio_weight
        )
    else:  # early fusion
        model = EarlyFusionModel(
            video_model_path=args.video_model,
            audio_model_path=args.audio_model
        )
    
    # Save the fusion model
    save_fusion_model(model, args.output_dir)
    print(f"Fusion model saved to {args.output_dir}")
    
    # Print usage instructions
    print("\nTo use this fusion model for inference:")
    print("1. Load both the video and audio models")
    print("2. Load the fusion configuration")
    print("3. Process video and audio inputs")
    print("4. Combine predictions according to the fusion type")
    
    if args.fusion_type == 'late':
        print("\nFor late fusion, you can adjust the weights at inference time if needed.")
    else:
        print("\nFor early fusion, you'll need to extract features from both models first.")
    
    print("\nSee the documentation for more details.")


if __name__ == '__main__':
    main()
