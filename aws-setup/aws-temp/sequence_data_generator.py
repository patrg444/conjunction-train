#!/usr/bin/env python3
"""
Custom data generator for sequence data that performs batch-wise padding.
This prevents information loss from truncation and reduces unnecessary padding.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence


class SequenceDataGenerator(Sequence):
    """
    Custom data generator for sequence data with batch-wise padding.
    
    This generator creates batches where sequences are only padded to the length
    of the longest sequence in that specific batch, rather than using a global
    maximum length. This approach:
    
    1. Prevents information loss by not truncating sequences
    2. Reduces unnecessary padding, making training more efficient
    3. Handles variable-length sequences more naturally
    
    The generator works with multimodal data (video and audio) and supports masking.
    """
    
    def __init__(self, video_features, audio_features, labels, 
                 batch_size=32, shuffle=True, masks=None):
        """
        Initialize the sequence data generator.
        
        Args:
            video_features: List of video feature arrays (variable length)
            audio_features: List of audio feature arrays (variable length)
            labels: Array of one-hot encoded labels
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the data after each epoch
            masks: Optional list of valid frame masks for the video features
        """
        self.video_features = video_features
        self.audio_features = audio_features
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.masks = masks
        
        # Store feature dimensions
        self.video_feature_dim = video_features[0].shape[1] if len(video_features[0].shape) > 1 else 1
        self.audio_feature_dim = audio_features[0].shape[1] if len(audio_features[0].shape) > 1 else 1
        
        # Create indices for accessing the data
        self.indices = np.arange(len(self.video_features))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        """Return the number of batches per epoch."""
        return int(np.ceil(len(self.video_features) / self.batch_size))
    
    def __getitem__(self, idx):
        """Get a batch of data with dynamic padding."""
        # Get batch indices
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Get batch data
        batch_video = [self.video_features[i] for i in batch_indices]
        batch_audio = [self.audio_features[i] for i in batch_indices]
        batch_labels = self.labels[batch_indices]
        
        # Get batch masks if available
        batch_masks = None
        if self.masks is not None:
            batch_masks = [self.masks[i] for i in batch_indices]
        
        # Compute maximum lengths for this batch
        max_video_length = max(len(video) for video in batch_video)
        max_audio_length = max(len(audio) for audio in batch_audio)
        
        # Create padded arrays for this batch
        batch_video_padded = np.zeros((len(batch_indices), max_video_length, self.video_feature_dim))
        batch_audio_padded = np.zeros((len(batch_indices), max_audio_length, self.audio_feature_dim))
        
        # Fill arrays with the actual data (with masking if provided)
        for i, (video_feat, audio_feat) in enumerate(zip(batch_video, batch_audio)):
            # Handle video features
            video_length = len(video_feat)
            
            # Apply mask if provided
            if batch_masks is not None:
                mask = batch_masks[i]
                for j in range(video_length):
                    if j < len(mask) and mask[j]:
                        batch_video_padded[i, j, :] = video_feat[j, :]
            else:
                # No mask, just copy the sequence
                batch_video_padded[i, :video_length, :] = video_feat
            
            # Handle audio features
            audio_length = len(audio_feat)
            batch_audio_padded[i, :audio_length, :] = audio_feat
        
        # Return the batch
        return [batch_video_padded, batch_audio_padded], batch_labels
    
    def on_epoch_end(self):
        """Shuffle indices after each epoch if shuffle is True."""
        if self.shuffle:
            np.random.shuffle(self.indices)


class ValidationDataGenerator(SequenceDataGenerator):
    """
    Custom data generator for validation data.
    
    Similar to SequenceDataGenerator but uses consistent padding across all batches
    to ensure consistent evaluation metrics.
    """
    
    def __init__(self, video_features, audio_features, labels, 
                 batch_size=32, masks=None):
        """
        Initialize the validation data generator.
        """
        super().__init__(video_features, audio_features, labels, 
                        batch_size=batch_size, shuffle=False, masks=masks)
        
        # Compute global maximum lengths for consistency
        self.max_video_length = max(len(video) for video in video_features)
        self.max_audio_length = max(len(audio) for audio in audio_features)
    
    def __getitem__(self, idx):
        """Get a batch of validation data with consistent padding."""
        # Get batch indices
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Get batch data
        batch_video = [self.video_features[i] for i in batch_indices]
        batch_audio = [self.audio_features[i] for i in batch_indices]
        batch_labels = self.labels[batch_indices]
        
        # Get batch masks if available
        batch_masks = None
        if self.masks is not None:
            batch_masks = [self.masks[i] for i in batch_indices]
        
        # Create padded arrays using consistent maximum lengths
        batch_video_padded = np.zeros((len(batch_indices), self.max_video_length, self.video_feature_dim))
        batch_audio_padded = np.zeros((len(batch_indices), self.max_audio_length, self.audio_feature_dim))
        
        # Fill arrays with the actual data (with masking if provided)
        for i, (video_feat, audio_feat) in enumerate(zip(batch_video, batch_audio)):
            # Handle video features
            video_length = len(video_feat)
            
            # Apply mask if provided
            if batch_masks is not None:
                mask = batch_masks[i]
                for j in range(video_length):
                    if j < len(mask) and mask[j]:
                        batch_video_padded[i, j, :] = video_feat[j, :]
            else:
                # No mask, just copy the sequence
                batch_video_padded[i, :video_length, :] = video_feat
            
            # Handle audio features
            audio_length = len(audio_feat)
            batch_audio_padded[i, :audio_length, :] = audio_feat
        
        # Return the batch
        return [batch_video_padded, batch_audio_padded], batch_labels
