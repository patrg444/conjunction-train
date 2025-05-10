#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

class VideoOnlyFacenetGenerator(Sequence):
    def __init__(self, video_feature_files, labels, batch_size=32, 
                 max_seq_len=None, shuffle=True):
        self.video_feature_files = video_feature_files
        self.labels = labels
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.video_feature_files))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return int(np.ceil(len(self.video_feature_files) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_video_files = [self.video_feature_files[i] for i in batch_indexes]
        batch_labels = [self.labels[i] for i in batch_indexes]
        
        # Load video features and pad sequences if necessary
        batch_video_features = []
        for file_path in batch_video_files:
            try:
                features = np.load(file_path)
                if self.max_seq_len:
                    if features.shape[0] > self.max_seq_len:
                        features = features[:self.max_seq_len]
                    elif features.shape[0] < self.max_seq_len:
                        padding = np.zeros((self.max_seq_len - features.shape[0], features.shape[1]))
                        features = np.vstack((features, padding))
                batch_video_features.append(features)
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
                # Return empty array with correct shape as fallback
                empty_features = np.zeros((self.max_seq_len or 1, 512))  # Facenet features are 512-dim
                batch_video_features.append(empty_features)
        
        return np.array(batch_video_features), np.array(batch_labels)
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.video_feature_files))
        if self.shuffle:
            np.random.shuffle(self.indexes)
