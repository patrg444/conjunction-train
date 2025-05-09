import os
import numpy as np
import torch
import logging # Import the logging module
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pytorch_lightning as pl

# Define logger at module level for use throughout the file
# logger = logging.getLogger(__name__) # Keep this, but also initialize in class

class MultimodalFusionDataset(Dataset):
    def __init__(self, merged_df, embedding_dir=None):
        """
        merged_df: DataFrame with columns:
            id, split, label, text_path, audio_path, video_path
        embedding_dir: Optional base directory for embeddings (if paths are relative)
        """
        self.df = merged_df.reset_index(drop=True)
        self.embedding_dir = embedding_dir

    def __len__(self):
        return len(self.df)

    def _full_path(self, rel_path):
        # Join only if rel_path is not absolute
        if self.embedding_dir and not os.path.isabs(rel_path):
            rel_path = os.path.join(self.embedding_dir, rel_path)
        # Collapse possible duplicate “embeddings/embeddings”
        rel_path = rel_path.replace("embeddings/embeddings", "embeddings")
        return rel_path

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Get embedding file paths
        text_path = self._full_path(row['text_path'])
        audio_path = self._full_path(row['audio_path'])
        video_path = self._full_path(row['video_path'])

        # Load embeddings
        text_emb = torch.from_numpy(np.load(text_path)).float()
        audio_emb = torch.from_numpy(np.load(audio_path)).float()
        video_emb = torch.from_numpy(np.load(video_path)).float()

        # Label: map to int if needed
        label = row['label']
        if isinstance(label, str):
            label = 1 if label.lower() in ['humor', 'laughter', 'funny', '1'] else 0
        label = int(label)
        label = torch.tensor(label, dtype=torch.long)

        return {
            'text_embedding': text_emb,
            'audio_embedding': audio_emb,
            'video_embedding': video_emb,
            'label': label,
            'id': row['id']
        }

class FusionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        manifest_path, # Changed to a single manifest_path
        batch_size=32,
        num_workers=4,
        embedding_dir=None # Kept for MultimodalFusionDataset, though paths in manifest should be absolute
    ):
        super().__init__()
        self.manifest_path = manifest_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.embedding_dir = embedding_dir # Passed to MultimodalFusionDataset
        self.logger = logging.getLogger(__name__) # Initialize logger in __init__

    def setup(self, stage=None):
        # Read the single comprehensive manifest
        # Use the class instance logger
        try:
            merged_df = pd.read_csv(self.manifest_path)
        except FileNotFoundError:
            self.logger.error(f"Manifest file not found: {self.manifest_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error reading manifest file {self.manifest_path}: {e}")
            raise
            
        # Ensure required columns are present
        required_cols = ['talk_id', 'text_embedding_path', 'audio_embedding_path', 'video_embedding_path', 'label', 'split']
        missing_cols = [col for col in required_cols if col not in merged_df.columns]
        if missing_cols:
            self.logger.error(f"Manifest {self.manifest_path} is missing required columns: {missing_cols}")
            raise ValueError(f"Manifest is missing columns: {missing_cols}")

        # Rename 'talk_id' to 'id' if 'id' is used by MultimodalFusionDataset internally
        # and 'text_embedding_path' to 'text_path' etc.
        # Based on MultimodalFusionDataset, it expects 'id', 'text_path', 'audio_path', 'video_path'
        merged_df = merged_df.rename(columns={
            'talk_id': 'id',
            'text_embedding_path': 'text_path',
            'audio_embedding_path': 'audio_path',
            'video_embedding_path': 'video_path'
        })
        
        # Save for later (optional, but can be useful for inspection)
        self.merged_df = merged_df

        # Split by 'split' column
        self.train_df = merged_df[merged_df['split'] == 'train']
        self.val_df   = merged_df[merged_df['split'] == 'val']
        self.test_df  = merged_df[merged_df['split'] == 'test']

        self.logger.info(f"Train samples: {len(self.train_df)}, Val samples: {len(self.val_df)}, Test samples: {len(self.test_df)}")

        self.train_dataset = MultimodalFusionDataset(self.train_df, self.embedding_dir)
        self.val_dataset   = MultimodalFusionDataset(self.val_df, self.embedding_dir)
        self.test_dataset  = MultimodalFusionDataset(self.test_df, self.embedding_dir)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
