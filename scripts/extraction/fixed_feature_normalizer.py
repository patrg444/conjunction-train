#!/usr/bin/env python
"""
Feature Normalization Module

This module provides normalization functions for both audio and video features
that match the exact normalization process used during training.
"""

import numpy as np
import os
import pickle

# Constants
NORMALIZATION_PATH_TEMPLATE = "models/dynamic_padding_no_leakage/{name}_normalization_stats.pkl"
# For backward compatibility
AUDIO_MEAN_STD_PATH = "models/dynamic_padding_no_leakage/audio_normalization_stats.pkl"

def normalize_features(features, mean=None, std=None, name="audio"):
    """
    Normalize features using the same method as in training.

    Args:
        features: Features to normalize (numpy array)
        mean: Optional pre-computed mean (if not provided uses estimated values)
        std: Optional pre-computed standard deviation
        name: Feature type ("audio" or "video")

    Returns:
        Normalized features
    """
    # If mean and std are not provided try to load saved values or use estimates
    if mean is None or std is None:
        mean, std = load_or_estimate_normalization_stats(name=name)

    # Avoid division by zero (same as in training)
    safe_std = np.where(std == 0, 1.0, std)

    # Apply normalization (exactly as in training)
    normalized = (features - mean) / safe_std

    return normalized

def load_or_estimate_normalization_stats(name="audio"):
    """
    Load saved normalization statistics or provide reasonable estimates.

    Args:
        name: Feature type ("audio" or "video")

    Returns:
        Tuple of (mean, std) for feature normalization
    """
    # Get the appropriate path for the feature type
    stats_path = NORMALIZATION_PATH_TEMPLATE.format(name=name)

    # Try to load saved normalization stats
    if os.path.exists(stats_path):
        try:
            with open(stats_path, 'rb') as f:
                stats = pickle.load(f)
                return stats['mean'], stats['std']
        except Exception as e:
            print(f"Error loading {name} normalization stats: {e}")

    # Fall back to reasonable estimates based on feature type
    if name == "audio":
        # Create estimated statistics for audio (89-dimensional eGeMAPS features)
        print(f"Using estimated normalization statistics for {name} features")
        est_mean = np.zeros((1, 89))  # Features are typically centered around 0
        est_std = np.ones((1, 89)) * 0.5  # Typical scale for normalized features
    elif name == "video":
        # Create estimated statistics for video (512-dimensional FaceNet embeddings)
        print(f"Using estimated normalization statistics for {name} features")
        est_mean = np.zeros((1, 512))  # FaceNet embeddings are roughly N(0,1)
        est_std = np.ones((1, 512))  # Standard deviation around 1.0
    else:
        # Generic fallback for unknown feature types
        print(f"Warning: Unknown feature type '{name}' using generic statistics")
        est_mean = np.zeros((1, 100))  # Generic placeholder
        est_std = np.ones((1, 100))

    return est_mean, est_std

# Add this function for backward compatibility
def load_normalization_stats(name="audio"):
    """
    Load normalization statistics from saved files.
    
    This is an alias for load_or_estimate_normalization_stats for backward compatibility.

    Args:
        name: Feature type ("audio" or "video")

    Returns:
        Tuple of (mean, std) for feature normalization
    """
    return load_or_estimate_normalization_stats(name=name)

def save_normalization_stats(mean, std, name="audio"):
    """
    Save normalization statistics for future use.

    Args:
        mean: Mean values for each feature dimension
        std: Standard deviation values for each feature dimension
        name: Feature type ("audio" or "video")
    """
    stats_path = NORMALIZATION_PATH_TEMPLATE.format(name=name)
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, 'wb') as f:
        pickle.dump({'mean': mean, 'std': std}, f)
    print(f"Saved {name} normalization statistics to {stats_path}")

def create_dummy_normalized_features(seq_length, feature_dim=89, name="audio"):
    """
    Create dummy normalized features for audio/video-only modes.

    Args:
        seq_length: Length of sequence to create
        feature_dim: Feature dimension (default: 89 for eGeMAPS or specify for video)
        name: Feature type ("audio" or "video")

    Returns:
        Normalized dummy features
    """
    # Create zeros but with slight noise to better mimic normalized features
    # This helps avoid the model seeing exact zeros which might cause issues
    dummy = np.random.normal(0, 0.01, (1, seq_length, feature_dim))
    return dummy
