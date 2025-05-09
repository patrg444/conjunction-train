#!/usr/bin/env python3
"""
Custom data generator with synchronized audio-visual augmentation for emotion recognition.
Extends the SequenceDataGenerator to apply augmentation techniques that preserve
temporal alignment between modalities.
"""

import numpy as np
import tensorflow as tf
from sequence_data_generator import SequenceDataGenerator, ValidationDataGenerator


class SynchronizedAugmentationDataGenerator(SequenceDataGenerator):
    """
    Data generator with synchronized audio-visual augmentation that preserves temporal alignment.
    
    This generator extends the SequenceDataGenerator with augmentation techniques
    that maintain synchronization between audio and visual features, critical for
    multimodal emotion recognition where timing relationships are essential.
    
    Augmentation techniques:
    1. Synchronized time stretching/compression - applies same factor to both modalities
    2. Correlated noise addition - adds scaled noise to both modalities
    3. Feature modification - applies consistent transformations that preserve emotion signals
    """
    
    def __init__(self, video_features, audio_features, labels, 
                 batch_size=32, shuffle=True, masks=None,
                 augmentation_factor=2.0, augmentation_probability=0.5):
        """
        Initialize the generator with augmentation parameters.
        
        Args:
            video_features: List of video feature arrays (variable length)
            audio_features: List of audio feature arrays (variable length)
            labels: Array of one-hot encoded labels
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the data after each epoch
            masks: Optional list of valid frame masks for the video features
            augmentation_factor: How much to increase the dataset size through augmentation
            augmentation_probability: Probability of applying augmentation to each sample
        """
        super().__init__(video_features, audio_features, labels, 
                          batch_size=batch_size, shuffle=shuffle, masks=masks)
        
        self.augmentation_factor = augmentation_factor
        self.augmentation_probability = augmentation_probability
        
        # Expand indices to account for augmentation
        original_indices = np.arange(len(self.video_features))
        augmented_indices = []
        
        # Create additional indices for augmented versions
        for idx in original_indices:
            # Original sample always included
            augmented_indices.append((idx, 0))  # (original_idx, augmentation_id)
            # Add augmented versions
            for aug_id in range(1, int(augmentation_factor)):
                augmented_indices.append((idx, aug_id))
        
        self.augmented_indices = np.array(augmented_indices)
        if self.shuffle:
            np.random.shuffle(self.augmented_indices)
        
        print(f"Created synchronized augmentation generator with:")
        print(f"- Original samples: {len(original_indices)}")
        print(f"- Augmented dataset size: {len(augmented_indices)}")
        print(f"- Augmentation factor: {augmentation_factor}")
        print(f"- Augmentation probability: {augmentation_probability}")
    
    def synchronized_augment(self, audio_features, video_features):
        """
        Apply synchronized augmentation to preserve temporal alignment.
        
        Args:
            audio_features: Original audio features array
            video_features: Original video features array
            
        Returns:
            Tuple of (augmented_audio, augmented_video) with synchronization preserved
        """
        # Choose augmentation type with weighted probabilities
        aug_type = np.random.choice(['time_warp', 'noise', 'both'], 
                                   p=[0.4, 0.3, 0.3])
        
        # Make copies to avoid modifying originals
        audio_result = audio_features.copy()
        video_result = video_features.copy()
        
        # 1. TIME WARPING - apply same factor to both modalities
        if aug_type in ['time_warp', 'both']:
            # Use a consistent stretch factor for both modalities
            stretch_factor = np.random.uniform(0.9, 1.1)  # 90-110% of original length
            
            # Get original lengths
            audio_len = len(audio_features)
            video_len = len(video_features)
            
            # Calculate new lengths maintaining their ratio
            new_audio_len = max(5, int(audio_len * stretch_factor))
            new_video_len = max(5, int(video_len * stretch_factor))
            
            # Create resampled feature arrays
            resampled_audio = np.zeros((new_audio_len, audio_features.shape[1]))
            resampled_video = np.zeros((new_video_len, video_features.shape[1]))
            
            # Create evenly spaced indices for sampling
            audio_indices = np.linspace(0, audio_len-1, new_audio_len)
            video_indices = np.linspace(0, video_len-1, new_video_len)
            
            # Interpolate features - process each feature dimension
            for i in range(audio_features.shape[1]):
                resampled_audio[:, i] = np.interp(audio_indices, 
                                               np.arange(audio_len), 
                                               audio_features[:, i])
            
            for i in range(video_features.shape[1]):
                resampled_video[:, i] = np.interp(video_indices, 
                                               np.arange(video_len), 
                                               video_features[:, i])
            
            # Update results
            audio_result = resampled_audio
            video_result = resampled_video
        
        # 2. NOISE ADDITION with correlation between modalities
        if aug_type in ['noise', 'both']:
            # Create correlated noise levels - essential for maintaining relationship
            # between modalities
            audio_noise_level = np.random.uniform(0.001, 0.01)
            
            # Scale video noise level based on relative standard deviations
            # This maintains consistent impact across modalities
            audio_std = np.std(audio_result)
            video_std = np.std(video_result)
            
            # Avoid division by zero
            if audio_std == 0:
                audio_std = 1.0
            if video_std == 0:
                video_std = 1.0
                
            video_noise_level = audio_noise_level * (video_std / audio_std)
            
            # Generate and apply noise
            audio_noise = np.random.normal(0, audio_noise_level, audio_result.shape)
            video_noise = np.random.normal(0, video_noise_level, video_result.shape)
            
            audio_result += audio_noise
            video_result += video_noise
        
        return audio_result, video_result
    
    def __len__(self):
        """Return the number of batches per epoch."""
        return int(np.ceil(len(self.augmented_indices) / self.batch_size))
    
    def __getitem__(self, idx):
        """Get a batch of data with dynamic padding and augmentation."""
        # Get batch indices (these are now tuples of (sample_idx, aug_id))
        batch_aug_indices = self.augmented_indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        if len(batch_aug_indices) == 0:
            # Handle empty batch case - return structure with minimal content
            empty_video = np.zeros((1, 1, self.video_feature_dim), dtype=np.float32)
            empty_audio = np.zeros((1, 1, self.audio_feature_dim), dtype=np.float32)
            empty_label = np.zeros((1, self.labels.shape[1]), dtype=np.float32)
            
            # Convert to TensorFlow tensors
            empty_video_tensor = tf.convert_to_tensor(empty_video, dtype=tf.float32)
            empty_audio_tensor = tf.convert_to_tensor(empty_audio, dtype=tf.float32)
            empty_label_tensor = tf.convert_to_tensor(empty_label, dtype=tf.float32)
            
            return {'video_input': empty_video_tensor, 'audio_input': empty_audio_tensor}, empty_label_tensor
        
        # Prepare lists for batch data
        batch_video = []
        batch_audio = []
        batch_indices = [i[0] for i in batch_aug_indices]  # Extract original indices
        batch_labels = self.labels[batch_indices]
        
        # Get batch masks if available
        batch_masks = None
        if self.masks is not None:
            batch_masks = [self.masks[i] for i in batch_indices]
        
        # Process each sample in the batch
        for i, (sample_idx, aug_id) in enumerate(batch_aug_indices):
            video_feat = self.video_features[sample_idx].copy()
            audio_feat = self.audio_features[sample_idx].copy()
            
            # Apply augmentation for non-zero aug_id with probability
            if aug_id > 0 and np.random.random() < self.augmentation_probability:
                audio_feat, video_feat = self.synchronized_augment(audio_feat, video_feat)
            
            batch_video.append(video_feat)
            batch_audio.append(audio_feat)
        
        # Compute maximum lengths for this batch
        max_video_length = max((len(video) for video in batch_video), default=1)
        max_audio_length = max((len(audio) for audio in batch_audio), default=1)
        
        # Create padded arrays for this batch
        batch_video_padded = np.zeros((len(batch_indices), max_video_length, self.video_feature_dim))
        batch_audio_padded = np.zeros((len(batch_indices), max_audio_length, self.audio_feature_dim))
        
        # Fill arrays with the actual data
        for i, (video_feat, audio_feat) in enumerate(zip(batch_video, batch_audio)):
            # Handle video features
            video_length = len(video_feat)
            batch_video_padded[i, :video_length, :] = video_feat
            
            # Handle audio features
            audio_length = len(audio_feat)
            batch_audio_padded[i, :audio_length, :] = audio_feat
        
        # Convert to TensorFlow tensors
        batch_video_tensor = tf.convert_to_tensor(batch_video_padded, dtype=tf.float32)
        batch_audio_tensor = tf.convert_to_tensor(batch_audio_padded, dtype=tf.float32)
        batch_labels_tensor = tf.convert_to_tensor(batch_labels, dtype=tf.float32)
        
        # Return the batch
        return {'video_input': batch_video_tensor, 'audio_input': batch_audio_tensor}, batch_labels_tensor
    
    def on_epoch_end(self):
        """Shuffle indices after each epoch if shuffle is True."""
        if self.shuffle:
            np.random.shuffle(self.augmented_indices)


class SynchronizedAugmentationValidationGenerator(ValidationDataGenerator):
    """
    Validation data generator that supports synchronized augmentation.
    
    This generator is similar to the ValidationDataGenerator but can optionally
    apply augmentation for validation-time augmentation techniques.
    """
    
    def __init__(self, video_features, audio_features, labels, 
                 batch_size=32, masks=None, apply_augmentation=False):
        """
        Initialize the validation data generator with optional augmentation.
        
        Args:
            apply_augmentation: Whether to apply augmentation during validation
                                (normally False, but can be True for specific evaluation)
        """
        super().__init__(video_features, audio_features, labels, 
                         batch_size=batch_size, masks=masks)
        
        self.apply_augmentation = apply_augmentation
