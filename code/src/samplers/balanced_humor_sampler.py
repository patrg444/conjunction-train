#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch Sampler that ensures each batch contains a balanced number
of laugh (positive) and non-laugh (negative) samples for the humor head.
"""

import torch
import numpy as np
from torch.utils.data import Sampler, Dataset

class BalancedHumorSampler(Sampler):
    """
    Samples elements such that each batch has approximately 50% laugh samples.

    Args:
        dataset (Dataset): The dataset to sample from. Assumes the dataset's
                           manifest or an accessible attribute provides label info.
        batch_size (int): The size of each batch. Must be an even number.
        label_key (str): The key in the dataset's manifest/data to find the 0/1 laugh label.
    """
    def __init__(self, dataset: Dataset, batch_size: int, label_key: str = 'label'):
        self.dataset = dataset
        self.batch_size = batch_size
        self.label_key = label_key

        if self.batch_size % 2 != 0:
            raise ValueError(f"batch_size must be even for 50/50 sampling, got {batch_size}")

        self.half_batch = batch_size // 2

        # Pre-fetch labels or indices for efficient sampling
        print("Initializing BalancedHumorSampler: Fetching labels...")
        try:
            # Attempt to get labels directly if dataset stores them (e.g., from manifest)
            labels = self.dataset.manifest[self.label_key].values
        except AttributeError:
            # Fallback: Iterate through dataset (slower) - Requires dataset.__getitem__ to be fast
            print("Warning: Manifest attribute not found, iterating through dataset for labels (can be slow).")
            labels = [self.dataset[i][self.label_key].item() for i in range(len(self.dataset))]
            labels = np.array(labels)
            
        self.positive_indices = np.where(labels == 1)[0]
        self.negative_indices = np.where(labels == 0)[0]

        if len(self.positive_indices) == 0 or len(self.negative_indices) == 0:
             raise ValueError("Dataset must contain both positive (laugh) and negative (non-laugh) samples.")
             
        print(f"  Found {len(self.positive_indices)} positive and {len(self.negative_indices)} negative samples.")

        # Calculate number of batches based on the smaller class to ensure balance
        self.num_batches = min(len(self.positive_indices), len(self.negative_indices)) // self.half_batch
        self.n_samples = self.num_batches * self.batch_size # Total samples per epoch

        if self.num_batches == 0:
             raise ValueError(f"Not enough samples in one class to form a single balanced batch of size {batch_size}.")
             
        print(f"  Will yield {self.num_batches} balanced batches per epoch ({self.n_samples} samples total).")


    def __iter__(self):
        # Shuffle indices for both classes at the start of each epoch
        np.random.shuffle(self.positive_indices)
        np.random.shuffle(self.negative_indices)

        # Create batches
        indices = []
        for i in range(self.num_batches):
            start_pos = i * self.half_batch
            start_neg = i * self.half_batch
            
            pos_batch = self.positive_indices[start_pos : start_pos + self.half_batch]
            neg_batch = self.negative_indices[start_neg : start_neg + self.half_batch]
            
            batch_indices = np.concatenate((pos_batch, neg_batch))
            np.random.shuffle(batch_indices) # Shuffle within the batch
            indices.extend(batch_indices.tolist())
            
        return iter(indices)

    def __len__(self):
        # Length is the total number of samples yielded per epoch
        return self.n_samples

    @property
    def epoch_pos_ratio(self):
        """Returns the ratio of positive samples in one epoch (should be ~0.5)."""
        # This is approximate if num_batches doesn't perfectly divide class sizes
        return (self.num_batches * self.half_batch) / self.n_samples if self.n_samples > 0 else 0


# Example Usage (requires a dataset object similar to HumorDataset)
if __name__ == '__main__':
    print("\nTesting BalancedHumorSampler...")

    # Create a dummy dataset class for testing
    class DummyHumorDataset(Dataset):
        def __init__(self, num_samples=1000, pos_ratio=0.2):
            self.num_samples = num_samples
            self.pos_ratio = pos_ratio
            self.labels = np.random.rand(num_samples) < pos_ratio
            self.manifest = pd.DataFrame({'label': self.labels.astype(int)}) # Simulate manifest access
            print(f"DummyDataset: {num_samples} samples, {int(self.labels.sum())} positive ({(100*self.pos_ratio):.1f}%)")

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # Return just the label for sampler testing
            return {'label': torch.tensor(self.labels[idx], dtype=torch.long)}

    dummy_dataset = DummyHumorDataset(num_samples=570, pos_ratio=0.3) # Example imbalance
    
    try:
        batch_size = 32
        sampler = BalancedHumorSampler(dummy_dataset, batch_size=batch_size)
        
        print(f"\nSampler length (total samples per epoch): {len(sampler)}")
        print(f"Expected positive ratio per epoch: {sampler.epoch_pos_ratio:.2f}")

        # Simulate one epoch
        epoch_indices = list(iter(sampler))
        print(f"First {batch_size*2} indices from epoch: {epoch_indices[:batch_size*2]}...")
        
        # Verify batch balance (approximately)
        epoch_labels = dummy_dataset.labels[epoch_indices]
        num_batches_yielded = len(epoch_indices) // batch_size
        
        print(f"\nSimulating {num_batches_yielded} batches:")
        for i in range(min(num_batches_yielded, 5)): # Check first 5 batches
             batch_indices = epoch_indices[i*batch_size : (i+1)*batch_size]
             batch_labels = dummy_dataset.labels[batch_indices]
             pos_in_batch = batch_labels.sum()
             print(f" Batch {i}: {pos_in_batch}/{batch_size} positive samples ({(100*pos_in_batch/batch_size):.1f}%)")

    except ValueError as e:
        print(f"Error initializing sampler: {e}")
    except Exception as e:
         print(f"An unexpected error occurred during testing: {e}")
