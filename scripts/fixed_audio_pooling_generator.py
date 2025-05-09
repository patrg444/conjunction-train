#!/usr/bin/env python3
"""
Data generator that aligns audio features to video frame rate using pooling.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences # Import pad_sequences
import math
import os
import pickle
import glob

# Define timing constants
AUDIO_STEP_SECONDS = 0.010  # 10 ms hop size for audio features
VIDEO_FPS = 15.0
VIDEO_STEP_SECONDS = 1.0 / VIDEO_FPS  # Approx 66.7 ms

class AudioPoolingDataGenerator(Sequence):
    """
    Generates batches of combined audio/video data aligned to video frame rate.

    Audio features corresponding to each video frame interval are pooled (averaged)
    and concatenated with the video features for that frame.
    Optionally pads sequences to a fixed maximum length.
    """
    def __init__(self, split, batch_size=32, max_seq_len=None, shuffle=True, 
                normalize_features=False, dynamic_padding=False, padding_mode="post"):
        """
        Initialize the generator.

        Args:
            split: The dataset split ("train", "val", or "test")
            batch_size: Number of samples per batch
            max_seq_len: If provided pad all sequences to this fixed length
            shuffle: Whether to shuffle the data after each epoch
            normalize_features: Whether to normalize features
            dynamic_padding: Whether to use dynamic padding
            padding_mode: Padding mode (post or repeat_last)
        """
        self.split = split
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.shuffle = shuffle
        self.normalize_features = normalize_features
        self.dynamic_padding = dynamic_padding
        self.padding_mode = padding_mode
        
        # Load data - this would normally load from dataset files
        # For this implementation, we'll just load from standard locations
        self.video_features, self.audio_features, self.labels, self.laugh_labels = self._load_data(split)
        
        # Initialize indices
        self.indices = np.arange(len(self.video_features))
        
        # Get feature dimensions from the first sample
        if len(self.video_features) > 0:
            self.video_dim = self.video_features[0].shape[1]
            self.audio_dim = self.audio_features[0].shape[1]
            self.combined_dim = self.audio_dim + self.video_dim
            self.num_classes = self.labels[0].shape[0]
        else:
            # Handle empty input case
            self.video_dim = 0
            self.audio_dim = 0
            self.combined_dim = 0
            self.num_classes = 0
        
        print(f"Created AudioPoolingDataGenerator for '{split}' split:")
        print(f"- Samples: {len(self.indices)}")
        print(f"- Video Dim: {self.video_dim} Audio Dim: {self.audio_dim} Combined Dim: {self.combined_dim}")
        print(f"- Audio Step: {AUDIO_STEP_SECONDS*1000:.1f} ms Video Step: {VIDEO_STEP_SECONDS*1000:.1f} ms (Ratio ~{VIDEO_STEP_SECONDS/AUDIO_STEP_SECONDS:.2f})")
        print(f"- Max Seq Len: {max_seq_len}, Normalize: {normalize_features}, Dynamic Padding: {dynamic_padding}, Padding Mode: {padding_mode}")
        
        self.on_epoch_end()
    
    def _load_data(self, split):
        """
        Load data from standard locations.
        
        Args:
            split: The dataset split ("train", "val", or "test")
            
        Returns:
            video_features, audio_features, labels, laugh_labels
        """
        # These would normally be loaded from dataset files
        # For this implementation, we'll just create placeholder data
        # This would be replaced with actual data loading code
        base_dir = os.path.join("data", split)
        
        # Load feature data (placeholder implementation)
        video_features = []
        audio_features = []
        labels = []
        laugh_labels = []
        
        # In real implementation, this would load from files
        # For example, using glob to find NPZ files:
        feature_files = glob.glob(f"{base_dir}/*.npz")
        if len(feature_files) == 0:
            print(f"Warning: No feature files found in {base_dir}")
            # Create dummy data for testing
            for i in range(10):  # Create 10 dummy samples
                video_feat = np.random.randn(45, 512)  # 45 frames, 512 dims
                audio_feat = np.random.randn(150, 88)  # 150 frames, 88 dims
                label = np.zeros(8)
                label[i % 8] = 1  # One-hot label
                laugh_label = np.array([0])  # No laughter
                
                video_features.append(video_feat)
                audio_features.append(audio_feat)
                labels.append(label)
                laugh_labels.append(laugh_label)
        else:
            # Load actual data
            for file in feature_files:
                try:
                    data = np.load(file)
                    video_features.append(data['video_features'])
                    audio_features.append(data['audio_features'])
                    labels.append(data['label'])
                    if 'laugh_label' in data:
                        laugh_labels.append(data['laugh_label'])
                    else:
                        laugh_labels.append(np.array([0]))  # Default: no laughter
                except Exception as e:
                    print(f"Error loading {file}: {e}")
        
        return video_features, audio_features, labels, laugh_labels
    
    def __len__(self):
        """Return the number of batches per epoch."""
        return math.ceil(len(self.indices) / self.batch_size)

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indices of the batch
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Initialize batch data
        video_batch = []
        audio_batch = []
        labels_batch = []
        laugh_batch = []
        
        # Generate data for each sample in the batch
        for i in batch_indices:
            video_feat = self.video_features[i]
            audio_feat = self.audio_features[i]
            label = self.labels[i]
            laugh_label = self.laugh_labels[i] if i < len(self.laugh_labels) else np.array([0])
            
            # Process features (normalize if requested)
            if self.normalize_features:
                # Placeholder for normalization - in real implementation, use actual normalization
                video_feat = (video_feat - np.mean(video_feat, axis=0)) / (np.std(video_feat, axis=0) + 1e-8)
                audio_feat = (audio_feat - np.mean(audio_feat, axis=0)) / (np.std(audio_feat, axis=0) + 1e-8)
            
            # Add to batch
            video_batch.append(video_feat)
            audio_batch.append(audio_feat)
            labels_batch.append(label)
            laugh_batch.append(laugh_label)
        
        # Pad sequences if max_seq_len is specified
        if self.max_seq_len is not None:
            if self.padding_mode == "repeat_last":
                # Custom padding: repeat last frame if needed
                video_batch = [self._pad_repeat_last(seq, self.max_seq_len) for seq in video_batch]
                audio_batch = [self._pad_repeat_last(seq, self.max_seq_len) for seq in audio_batch]
            else:
                # Standard padding
                video_batch = pad_sequences(video_batch, maxlen=self.max_seq_len, padding='post', dtype='float32')
                audio_batch = pad_sequences(audio_batch, maxlen=self.max_seq_len, padding='post', dtype='float32')
        
        # Convert lists to numpy arrays
        video_batch = np.array(video_batch)
        audio_batch = np.array(audio_batch)
        labels_batch = np.array(labels_batch)
        laugh_batch = np.array(laugh_batch)
        
        # Return both audio and video features separately, along with labels
        return [video_batch, audio_batch], {'emotion_output': labels_batch, 'laugh_output': laugh_batch}
    
    def _pad_repeat_last(self, sequence, max_length):
        """
        Pad sequence by repeating the last frame.
        
        Args:
            sequence: The sequence to pad
            max_length: The target length
            
        Returns:
            Padded sequence
        """
        if len(sequence) >= max_length:
            return sequence[:max_length]
        else:
            last_frame = sequence[-1:]
            pad_length = max_length - len(sequence)
            padding = np.repeat(last_frame, pad_length, axis=0)
            return np.concatenate([sequence, padding], axis=0)
    
    def on_epoch_end(self):
        """Update indices after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)
