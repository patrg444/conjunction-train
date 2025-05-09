#!/usr/bin/env python3
"""
Data generator that loads precomputed audio/video features by split,
pads/truncates sequences to a fixed length, and yields batches suitable
for the dual-input emotion model.
"""

import os
import glob
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

class AudioPoolingDataGenerator(Sequence):
    """
    Generates batches of audio and video feature sequences for emotion recognition.

    Args:
        split: Dataset split ("train", "val", or "test")
        batch_size: Number of samples per batch
        max_seq_len: Pad or truncate all sequences to this length
        shuffle: Whether to shuffle sample order each epoch
        normalize_features: If True, apply feature-wise normalization per sequence
        dynamic_padding: (unused) placeholder for compatibility
        padding_mode: "post" or "repeat_last" behavior when padding shorter sequences
    """
    def __init__(self,
                 split,
                 batch_size=32,
                 max_seq_len=None,
                 shuffle=True,
                 normalize_features=False,
                 dynamic_padding=False,
                 padding_mode="post"):
        super().__init__()
        self.split = split
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.shuffle = shuffle
        self.normalize_features = normalize_features
        self.padding_mode = padding_mode

        base_dir = os.path.join("data", split)
        files = glob.glob(os.path.join(base_dir, "*.npz"))
        if not files:
            raise ValueError(f"No feature files found in {base_dir}")

        self.video_features = []
        self.audio_features = []
        self.labels = []

        for fp in files:
            data = np.load(fp)
            self.video_features.append(data["video_features"])
            self.audio_features.append(data["audio_features"])
            self.labels.append(data["label"])

        self.indices = np.arange(len(self.video_features))
        if self.video_features:
            self.video_dim = self.video_features[0].shape[1]
            self.audio_dim = self.audio_features[0].shape[1]
        else:
            self.video_dim = 0
            self.audio_dim = 0

        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)

    def __getitem__(self, idx):
        batch_ids = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        audio_batch = []
        video_batch = []
        label_batch = []
        for i in batch_ids:
            v = self.video_features[i]
            a = self.audio_features[i]
            y = self.labels[i]
            if self.normalize_features:
                v = (v - np.mean(v, axis=0)) / (np.std(v, axis=0) + 1e-8)
                a = (a - np.mean(a, axis=0)) / (np.std(a, axis=0) + 1e-8)
            if self.max_seq_len is not None:
                v = self._pad_or_truncate(v, self.max_seq_len)
                a = self._pad_or_truncate(a, self.max_seq_len)
            audio_batch.append(a)
            video_batch.append(v)
            label_batch.append(y)
        audio_batch = np.array(audio_batch, dtype=np.float32)
        video_batch = np.array(video_batch, dtype=np.float32)
        label_batch = np.array(label_batch, dtype=np.float32)
        print("DEBUG: __getitem__ returning", type(audio_batch), type(video_batch))
        return {"audio_input": audio_batch, "video_input": video_batch}, label_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _pad_or_truncate(self, seq, length):
        if seq.shape[0] == length:
            return seq
        if seq.shape[0] > length:
            return seq[:length]
        if self.padding_mode == "repeat_last":
            last = seq[-1:]
            repeats = np.repeat(last, length - seq.shape[0], axis=0)
            return np.vstack([seq, repeats])
        pad_width = ((0, length - seq.shape[0]), (0, 0))
        return np.pad(seq, pad_width, mode="constant")
