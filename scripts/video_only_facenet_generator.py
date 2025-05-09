#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data generator for video-only emotion recognition model using Facenet features.
This generator handles loading facial embeddings from .npz files and preparing them for LSTM training.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import math
from tqdm import tqdm

class VideoOnlyFacenetGenerator(Sequence):
    """
    Generates batches of Facenet facial embedding features for video-only mode.
    Handles loading, padding, and preparing features for LSTM model training.
    """
    def __init__(self, video_feature_files, labels, batch_size=32, shuffle=True, max_seq_len=None, audio_mode='video_only'):
        self.video_files = video_feature_files
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_seq_len = max_seq_len
        self.audio_mode = audio_mode  # 'video_only' is expected
        
        # Validate mode
        if self.audio_mode != 'video_only':
            print(f"Warning: Expected audio_mode='video_only', got '{self.audio_mode}'. Proceeding with video-only mode.")
        
        self.indices = np.arange(len(self.video_files))
        self.video_feature_dim = 0
        
        # Determine video feature dimension from the first valid file
        print("Determining video feature dimensions...")
        for i in range(min(10, len(self.video_files))):
            try:
                with np.load(self.video_files[i], allow_pickle=True) as data:
                    if 'video_features' in data:
                        features = data['video_features']
                        if features.shape[0] > 0 and features.shape[1] > 0:
                            self.video_feature_dim = features.shape[1]
                            print(f"Found video feature dimension: {self.video_feature_dim}")
                            break
            except Exception as e:
                print(f"Warning: Error loading file {i}: {e}")
                continue
                
        if self.video_feature_dim == 0:
            print("Warning: Could not determine video feature dimension. Using default of 512.")
            self.video_feature_dim = 512  # Default to standard Facenet dimension
            
        print(f"VideoOnlyFacenetGenerator:")
        print(f"- Samples: {len(self.indices)}")
        print(f"- Video Feature Dimension: {self.video_feature_dim}")
        print(f"- Max Sequence Length: {self.max_seq_len if self.max_seq_len else 'Dynamic'}")
        print(f"- Mode: {self.audio_mode}")
        
        self.on_epoch_end()
        
    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)
        
    def __getitem__(self, index):
        # Get indices for this batch
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Initialize lists to store batch data
        batch_features = []
        batch_labels = []
        
        # Process each sample in the batch
        for i in batch_indices:
            try:
                # Load video features from npz file with allow_pickle=True
                with np.load(self.video_files[i], allow_pickle=True) as data:
                    if 'video_features' in data:
                        video_features = data['video_features'].astype(np.float32)
                        if video_features.shape[0] > 0:
                            batch_features.append(video_features)
                            batch_labels.append(self.labels[i])
                        else:
                            print(f"Warning: Empty video features in {self.video_files[i]}")
                    else:
                        print(f"Warning: No 'video_features' key in {self.video_files[i]}")
            except Exception as e:
                print(f"Error loading {self.video_files[i]}: {e}")
                continue
                
        # Handle empty batch case
        if not batch_features:
            # Return empty tensors with correct shapes
            empty_features = np.zeros((0, self.max_seq_len if self.max_seq_len else 1, self.video_feature_dim), dtype=np.float32)
            empty_labels = np.zeros((0, self.labels[0].shape[0] if len(self.labels) > 0 else 6), dtype=np.float32)
            return tf.convert_to_tensor(empty_features), tf.convert_to_tensor(empty_labels)
            
        # Pad sequences to maximum length in batch or predefined max_seq_len
        if self.max_seq_len is not None:
            batch_padded = pad_sequences(batch_features, maxlen=self.max_seq_len, dtype='float32', padding='post', truncating='post')
        else:
            max_len_in_batch = max(len(seq) for seq in batch_features)
            batch_padded = pad_sequences(batch_features, maxlen=max_len_in_batch, dtype='float32', padding='post', truncating='post')
            
        # Convert to TensorFlow tensors
        batch_padded_tensor = tf.convert_to_tensor(batch_padded, dtype=tf.float32)
        batch_labels_tensor = tf.convert_to_tensor(np.array(batch_labels), dtype=tf.float32)
        
        return batch_padded_tensor, batch_labels_tensor
        
    def on_epoch_end(self):
        """Shuffle indices at the end of each epoch if shuffle=True"""
        if self.shuffle:
            np.random.shuffle(self.indices)

# Example usage (if run directly)
if __name__ == '__main__':
    print("Video-Only Facenet Generator Example (requires actual data)")
    # This section executes if the script is run directly
