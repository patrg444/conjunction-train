#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed Video Facenet Generator

This generator properly loads Facenet feature files and handles
sequence data for LSTM training with proper normalization.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

class FixedVideoFacenetGenerator(Sequence):
    """
    Generator for video-only Facenet features with fixed loading
    to handle variable-length sequences and proper normalization.
    """
    
    def __init__(self, video_feature_files, labels=None, batch_size=32, 
                 shuffle=True, normalize_features=True, max_seq_len=None,
                 test_mode=False):
        """
        Initialize the generator with feature files and labels.
        
        Args:
            video_feature_files (list): List of paths to video feature files
            labels (np.ndarray, optional): Array of labels for each file
            batch_size (int): Batch size for training
            shuffle (bool): Whether to shuffle the data
            normalize_features (bool): Whether to normalize features
            max_seq_len (int, optional): Maximum sequence length
            test_mode (bool): Whether to use test mode
        """
        self.video_feature_files = video_feature_files
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.normalize_features = normalize_features
        self.test_mode = test_mode
        
        # Determine max sequence length if not provided
        self._max_seq_len = max_seq_len
        if self._max_seq_len is None:
            self._max_seq_len = self._determine_max_seq_len()
        
        # Compute mean and std for normalization
        if normalize_features:
            self.feature_mean, self.feature_std = self._compute_normalization_stats()
        
        # Create indices for shuffling
        self.indices = np.arange(len(self.video_feature_files))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def _determine_max_seq_len(self, sample_size=100):
        """Determine maximum sequence length from a sample of files."""
        max_len = 0
        sample_size = min(sample_size, len(self.video_feature_files))
        sample_indices = np.random.choice(len(self.video_feature_files), sample_size, replace=False)
        
        for idx in sample_indices:
            try:
                file_path = self.video_feature_files[idx]
                with np.load(file_path, allow_pickle=True) as data:
                    if 'video_features' in data:
                        features = data['video_features']
                        if features.shape[0] > max_len:
                            max_len = features.shape[0]
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return max_len
    
    def _compute_normalization_stats(self, sample_size=500):
        """Compute mean and std for normalization from a sample of files."""
        sample_size = min(sample_size, len(self.video_feature_files))
        sample_indices = np.random.choice(len(self.video_feature_files), sample_size, replace=False)
        
        # Collect samples for statistics calculation
        all_features = []
        for idx in sample_indices:
            try:
                file_path = self.video_feature_files[idx]
                with np.load(file_path, allow_pickle=True) as data:
                    if 'video_features' in data:
                        features = data['video_features']
                        # Sample a few frames from each video to avoid bias
                        if features.shape[0] > 10:
                            sample_frames = np.random.choice(features.shape[0], 10, replace=False)
                            all_features.append(features[sample_frames])
                        else:
                            all_features.append(features)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        if not all_features:
            print("Warning: No features found for normalization")
            return 0.0, 1.0
        
        # Stack all collected features
        all_features = np.vstack(all_features)
        
        # Compute mean and std
        feature_mean = np.mean(all_features)
        feature_std = np.std(all_features)
        
        # Avoid division by zero
        if feature_std < 1e-10:
            feature_std = 1.0
        
        return feature_mean, feature_std
    
    def __len__(self):
        """Return the number of batches per epoch."""
        return int(np.ceil(len(self.video_feature_files) / self.batch_size))
    
    def __getitem__(self, idx):
        """Get a batch of data."""
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x, batch_y = self._generate_batch(batch_indices)
        
        if self.labels is None:
            return batch_x
        else:
            return batch_x, batch_y
    
    def _generate_batch(self, batch_indices):
        """Generate a batch of data."""
        # Initialize batch arrays
        batch_x = np.zeros((len(batch_indices), self._max_seq_len, 512))
        
        # For classification tasks
        if self.labels is not None:
            batch_y = np.zeros(len(batch_indices))
        else:
            batch_y = None
        
        # Fill batch arrays
        for i, idx in enumerate(batch_indices):
            try:
                # Load feature file
                file_path = self.video_feature_files[idx]
                with np.load(file_path, allow_pickle=True) as data:
                    # Get video features
                    if 'video_features' in data:
                        features = data['video_features']
                        seq_len = min(features.shape[0], self._max_seq_len)
                        
                        # Normalize features if enabled
                        if self.normalize_features:
                            features = (features - self.feature_mean) / self.feature_std
                        
                        # Add to batch
                        batch_x[i, :seq_len, :] = features[:seq_len]
                    
                    # Add label if available
                    if self.labels is not None:
                        batch_y[i] = self.labels[idx]
            
            except Exception as e:
                print(f"Error generating batch for {self.video_feature_files[idx]}: {e}")
        
        return batch_x, batch_y
    
    def on_epoch_end(self):
        """Called at the end of each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def get_max_sequence_length(self):
        """Return the maximum sequence length."""
        return self._max_seq_len
    
    def split_train_val(self, val_ratio=0.2, seed=42):
        """Split the generator into training and validation generators."""
        np.random.seed(seed)
        indices = np.arange(len(self.video_feature_files))
        np.random.shuffle(indices)
        
        split_idx = int(len(indices) * (1 - val_ratio))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_files = [self.video_feature_files[i] for i in train_indices]
        val_files = [self.video_feature_files[i] for i in val_indices]
        
        if self.labels is not None:
            train_labels = self.labels[train_indices]
            val_labels = self.labels[val_indices]
        else:
            train_labels = None
            val_labels = None
        
        train_gen = FixedVideoFacenetGenerator(
            video_feature_files=train_files,
            labels=train_labels,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            normalize_features=self.normalize_features,
            max_seq_len=self._max_seq_len
        )
        
        val_gen = FixedVideoFacenetGenerator(
            video_feature_files=val_files,
            labels=val_labels,
            batch_size=self.batch_size,
            shuffle=False,  # No shuffle for validation
            normalize_features=self.normalize_features,
            max_seq_len=self._max_seq_len
        )
        
        return train_gen, val_gen

if __name__ == '__main__':
    # Example usage
    video_files = ['/path/to/video/features.npz']
    labels = np.array([0])  # Example label
    
    gen = FixedVideoFacenetGenerator(
        video_feature_files=video_files,
        labels=labels,
        batch_size=32,
        shuffle=True,
        normalize_features=True
    )
    
    print(f"Generator length: {len(gen)}")
    print(f"Max sequence length: {gen.get_max_sequence_length()}")
