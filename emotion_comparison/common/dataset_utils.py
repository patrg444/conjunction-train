#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset utilities for emotion recognition feature extraction comparison.
Handles loading, preprocessing, and splitting datasets with consistent emotion mapping.
"""

import os
import glob
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import json
import tensorflow as tf
from tqdm import tqdm

# Constants for emotion mapping
EMOTION_NAMES = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness']
EMOTION_SHORT = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']  # Short codes
EMOTION_MAP = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}  # Index mapping

# RAVDESS specific mapping from filename emotion codes to our standardized format
RAVDESS_EMOTION_MAP = {
    '01': 'NEU',  # neutral
    '02': 'NEU',  # calm (mapped to neutral)
    '03': 'HAP',  # happy
    '04': 'SAD',  # sad
    '05': 'ANG',  # angry
    '06': 'FEA',  # fearful
    '07': 'DIS',  # disgust
    '08': 'FEA'   # surprised (mapped to fear)
}

class EmotionDatasetManager:
    """
    Manages emotion recognition datasets (RAVDESS and CREMA-D) with consistent emotion mapping,
    train/val/test splitting, and various utility functions.
    """
    
    def __init__(self, ravdess_dir='./downsampled_videos/RAVDESS',
                 cremad_dir='./downsampled_videos/CREMA-D-audio-complete',
                 output_dir='./processed_data',
                 video_ext=None):
        """
        Initialize the dataset manager.
        
        Args:
            ravdess_dir: Path to RAVDESS dataset directory
            cremad_dir: Path to CREMA-D dataset directory
            output_dir: Directory to save processed data
            video_ext: Video file extension
        """
        self.ravdess_dir = ravdess_dir
        self.cremad_dir = cremad_dir
        self.output_dir = output_dir
        self.video_ext = video_ext
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all video files
        self.ravdess_files = self._find_ravdess_files()
        self.cremad_files = self._find_cremad_files()
        
        # Initialize splits
        self.train_files = []
        self.val_files = []
        self.test_files = []
        
        print(f"Found {len(self.ravdess_files)} RAVDESS files")
        print(f"Found {len(self.cremad_files)} CREMA-D files")
    
    def _find_ravdess_files(self):
        """Find all RAVDESS video files - using .mp4 extension by default."""
        if self.video_ext:
            pattern = os.path.join(self.ravdess_dir, "**", f"*{self.video_ext}")
        else:
            pattern = os.path.join(self.ravdess_dir, "**", "*.mp4")
        return glob.glob(pattern, recursive=True)
    
    def _find_cremad_files(self):
        """Find all CREMA-D video files - looking for both .flv and .mp4 files."""
        files = []
        if self.video_ext:
            pattern = os.path.join(self.cremad_dir, f"*{self.video_ext}")
            files = glob.glob(pattern)
        else:
            # Look for both .flv and .mp4 files (CREMA-D dataset has .flv files)
            flv_files = glob.glob(os.path.join(self.cremad_dir, "*.flv"))
            mp4_files = glob.glob(os.path.join(self.cremad_dir, "*.mp4"))
            files = flv_files + mp4_files
        return files
    
    def get_emotion_from_path(self, file_path):
        """
        Extract emotion label from file path.
        
        Args:
            file_path: Path to video file
            
        Returns:
            emotion_idx: Integer emotion index (0-5)
            emotion_name: String emotion name
        """
        basename = os.path.basename(file_path)
        basename = os.path.splitext(basename)[0]  # Remove extension
        
        # RAVDESS format: 01-01-03-02-01-01-XX.mp4 (3rd position is emotion)
        if "Actor_" in file_path:
            parts = basename.split('-')
            if len(parts) >= 3:
                emotion_code = parts[2]
                emotion_short = RAVDESS_EMOTION_MAP.get(emotion_code, 'UNK')
            else:
                return None, None
        
        # CREMA-D format: 1076_IEO_ANG_XX.mp4
        else:
            parts = basename.split('_')
            if len(parts) >= 3:
                emotion_short = parts[2]
            else:
                return None, None
        
        # Map to our standard emotion indices
        if emotion_short in EMOTION_MAP:
            emotion_idx = EMOTION_MAP[emotion_short]
            emotion_name = EMOTION_NAMES[emotion_idx]
            return emotion_idx, emotion_name
        else:
            return None, None
    
    def analyze_class_distribution(self):
        """
        Analyze emotion class distribution across datasets.
        
        Returns:
            ravdess_counts: Counter for RAVDESS emotions
            cremad_counts: Counter for CREMA-D emotions
            combined_counts: Counter for combined emotions
        """
        ravdess_counts = Counter()
        cremad_counts = Counter()
        combined_counts = Counter()
        
        # Process RAVDESS files
        valid_ravdess = 0
        for file_path in tqdm(self.ravdess_files, desc="Processing RAVDESS"):
            emotion_idx, emotion_name = self.get_emotion_from_path(file_path)
            if emotion_idx is not None:
                emotion_short = EMOTION_SHORT[emotion_idx]
                ravdess_counts[emotion_short] += 1
                combined_counts[emotion_short] += 1
                valid_ravdess += 1
        
        # Process CREMA-D files
        valid_cremad = 0
        for file_path in tqdm(self.cremad_files, desc="Processing CREMA-D"):
            emotion_idx, emotion_name = self.get_emotion_from_path(file_path)
            if emotion_idx is not None:
                emotion_short = EMOTION_SHORT[emotion_idx]
                cremad_counts[emotion_short] += 1
                combined_counts[emotion_short] += 1
                valid_cremad += 1
        
        # Print statistics
        print(f"\nRAVDESS valid files: {valid_ravdess} / {len(self.ravdess_files)}")
        print(f"CREMA-D valid files: {valid_cremad} / {len(self.cremad_files)}")
        print(f"Total valid files: {valid_ravdess + valid_cremad}")
        
        print("\nEmotion Distribution:")
        total = sum(combined_counts.values())
        for emotion in EMOTION_SHORT:
            count = combined_counts[emotion]
            print(f"{emotion}: {count} ({count/total*100:.1f}%)")
        
        # Calculate class weights
        class_weights = self.calculate_class_weights(combined_counts)
        
        return ravdess_counts, cremad_counts, combined_counts, class_weights
    
    def calculate_class_weights(self, combined_counts):
        """
        Calculate class weights for balanced training.
        
        Args:
            combined_counts: Counter with emotion counts
            
        Returns:
            class_weights: Dictionary of class weights
        """
        # Get total samples per class
        counts = np.array([combined_counts.get(emotion, 0) for emotion in EMOTION_SHORT])
        
        # Compute class weights: n_samples / (n_classes * np.bincount(y))
        n_samples = np.sum(counts)
        n_classes = len(EMOTION_MAP)
        
        class_weights = {}
        for emotion, idx in sorted(EMOTION_MAP.items(), key=lambda x: x[1]):
            count = combined_counts.get(emotion, 0)
            if count > 0:
                weight = n_samples / (n_classes * count)
                class_weights[idx] = weight
            else:
                class_weights[idx] = 1.0
        
        print("\nClass Weights for Training:")
        for idx, weight in sorted(class_weights.items()):
            emotion = EMOTION_SHORT[idx]
            print(f"{emotion} (class {idx}): {weight:.4f}")
        
        return class_weights
    
    def create_dataset_splits(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, 
                              stratify=True, random_state=42):
        """
        Create train/validation/test splits for the datasets.
        
        Args:
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            test_ratio: Fraction of data for testing
            stratify: Whether to stratify splits by emotion
            random_state: Random seed for reproducibility
            
        Returns:
            train_files, val_files, test_files: Lists of file paths for each split
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
        
        # Combine datasets
        all_files = []
        all_labels = []
        
        # Process RAVDESS
        for file_path in self.ravdess_files:
            emotion_idx, _ = self.get_emotion_from_path(file_path)
            if emotion_idx is not None:
                all_files.append(file_path)
                all_labels.append(emotion_idx)
        
        # Process CREMA-D
        for file_path in self.cremad_files:
            emotion_idx, _ = self.get_emotion_from_path(file_path)
            if emotion_idx is not None:
                all_files.append(file_path)
                all_labels.append(emotion_idx)
        
        # Convert to numpy arrays
        all_files = np.array(all_files)
        all_labels = np.array(all_labels)
        
        # First split: train + validation vs test
        stratify_param = all_labels if stratify else None
        train_val_files, test_files, train_val_labels, test_labels = train_test_split(
            all_files, all_labels, 
            test_size=test_ratio, 
            random_state=random_state,
            stratify=stratify_param
        )
        
        # Second split: train vs validation
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        stratify_param = train_val_labels if stratify else None
        train_files, val_files, train_labels, val_labels = train_test_split(
            train_val_files, train_val_labels,
            test_size=val_ratio_adjusted,
            random_state=random_state,
            stratify=stratify_param
        )
        
        # Store splits
        self.train_files = train_files.tolist()
        self.val_files = val_files.tolist()
        self.test_files = test_files.tolist()
        
        # Print split statistics
        print(f"\nDataset Split Statistics:")
        print(f"Training set: {len(train_files)} files")
        print(f"Validation set: {len(val_files)} files")
        print(f"Test set: {len(test_files)} files")
        
        # Save splits to disk
        self._save_splits()
        
        # Verify class distribution in splits
        self._verify_split_distribution(train_labels, val_labels, test_labels)
        
        return self.train_files, self.val_files, self.test_files
    
    def _save_splits(self):
        """Save dataset splits to disk."""
        splits = {
            'train': self.train_files,
            'validation': self.val_files,
            'test': self.test_files
        }
        
        split_file = os.path.join(self.output_dir, 'dataset_splits.json')
        with open(split_file, 'w') as f:
            json.dump(splits, f, indent=2)
        
        print(f"Dataset splits saved to {split_file}")
    
    def _verify_split_distribution(self, train_labels, val_labels, test_labels):
        """Verify emotion distribution across splits."""
        train_dist = Counter(train_labels)
        val_dist = Counter(val_labels)
        test_dist = Counter(test_labels)
        
        print("\nEmotion Distribution in Splits:")
        for i, emotion in enumerate(EMOTION_NAMES):
            train_count = train_dist.get(i, 0)
            val_count = val_dist.get(i, 0)
            test_count = test_dist.get(i, 0)
            total = train_count + val_count + test_count
            
            print(f"{emotion}: Train {train_count} ({train_count/total*100:.1f}%), "
                  f"Val {val_count} ({val_count/total*100:.1f}%), "
                  f"Test {test_count} ({test_count/total*100:.1f}%)")
    
    def load_splits(self):
        """Load dataset splits from disk if available."""
        split_file = os.path.join(self.output_dir, 'dataset_splits.json')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                splits = json.load(f)
            
            self.train_files = splits['train']
            self.val_files = splits['validation']
            self.test_files = splits['test']
            
            print(f"Loaded dataset splits from {split_file}")
            print(f"Training set: {len(self.train_files)} files")
            print(f"Validation set: {len(self.val_files)} files")
            print(f"Test set: {len(self.test_files)} files")
            
            return True
        else:
            print(f"No saved splits found at {split_file}")
            return False
    
    def visualize_distributions(self, ravdess_counts, cremad_counts, combined_counts):
        """
        Create a bar chart comparing the distributions.
        
        Args:
            ravdess_counts: Counter for RAVDESS emotions
            cremad_counts: Counter for CREMA-D emotions
            combined_counts: Counter for combined emotions
        """
        # Prepare data
        emotions = EMOTION_SHORT
        ravdess_values = [ravdess_counts.get(e, 0) for e in emotions]
        cremad_values = [cremad_counts.get(e, 0) for e in emotions]
        combined_values = [combined_counts.get(e, 0) for e in emotions]
        
        # Set up the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set the width of the bars
        bar_width = 0.25
        index = np.arange(len(emotions))
        
        # Create the bars
        ravdess_bars = ax.bar(index - bar_width, ravdess_values, bar_width, 
                            label='RAVDESS', color='skyblue', edgecolor='black')
        cremad_bars = ax.bar(index, cremad_values, bar_width,
                           label='CREMA-D', color='lightgreen', edgecolor='black')
        combined_bars = ax.bar(index + bar_width, combined_values, bar_width,
                             label='Combined', color='salmon', edgecolor='black')
        
        # Add labels, title, etc.
        ax.set_xlabel('Emotion', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        ax.set_title('Emotion Distribution Across Datasets', fontsize=16, fontweight='bold')
        ax.set_xticks(index)
        ax.set_xticklabels(emotions)
        ax.legend()
        
        # Add count labels on top of each bar
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{int(height)}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
                            
        add_labels(ravdess_bars)
        add_labels(cremad_bars)
        add_labels(combined_bars)
        
        # Calculate and add class imbalance ratio
        counts = list(combined_counts.values())
        min_count = min(counts) if counts else 0
        max_count = max(counts) if counts else 0
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        ax.text(0.5, 0.95, f'Class Imbalance Ratio: {imbalance_ratio:.2f}', 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax.transAxes, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.7))
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'emotion_distribution.png'))
        print(f"\nVisualization saved to {os.path.join(self.output_dir, 'emotion_distribution.png')}")
        
        return fig
    
    @staticmethod
    def get_frame_count(video_path):
        """
        Get the number of frames in a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            frame_count: Number of frames in the video
        """
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count
    
    @staticmethod
    def extract_frames(video_path, max_frames=None, skip_frames=1):
        """
        Extract frames from a video file.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract (None for all)
            skip_frames: Number of frames to skip between extractions
            
        Returns:
            frames: List of extracted frames
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if (frame_count - 1) % skip_frames == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            if max_frames is not None and len(frames) >= max_frames:
                break
        
        cap.release()
        return frames
    
    @staticmethod
    def create_tf_dataset(file_list, batch_size=32, shuffle=True, augment=False, 
                          prefetch=tf.data.AUTOTUNE, feature_extractor=None):
        """
        Create a TensorFlow dataset from a list of video files.
        
        Args:
            file_list: List of video file paths
            batch_size: Batch size for the dataset
            shuffle: Whether to shuffle the dataset
            augment: Whether to apply data augmentation
            prefetch: Prefetch buffer size
            feature_extractor: Function to extract features from frames
            
        Returns:
            dataset: TensorFlow dataset
        """
        # This is a placeholder - actual implementation depends on the feature extraction method
        # and will be specialized in each method's implementation
        pass


def test_dataset_manager():
    """Test the dataset manager functionality."""
    manager = EmotionDatasetManager()
    
    # Analyze class distribution
    r_counts, c_counts, combined, weights = manager.analyze_class_distribution()
    
    # Visualize distributions
    manager.visualize_distributions(r_counts, c_counts, combined)
    
    # Create dataset splits
    manager.create_dataset_splits()


if __name__ == "__main__":
    test_dataset_manager()
