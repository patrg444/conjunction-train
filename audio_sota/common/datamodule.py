import os
import pytorch_lightning as pl
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
import pandas as pd
import numpy as np
from typing import Optional, List, Union, Dict, Tuple
from collections import Counter

# Assuming augmentations are defined in a sibling file/module
# from .augment import build_augmentations 

class AudioDataset(Dataset):
    """Dataset to load raw audio waveforms, resample, pad/truncate, and apply transforms."""
    def __init__(self, 
                 metadata_csv: str, 
                 audio_base_dir: str, 
                 target_sr: int = 16000, 
                 max_length_samples: Optional[int] = None, 
                 transforms=None,
                 label_map: Optional[Dict[str, int]] = None):
        """
        Args:
            metadata_csv (str): Path to the split CSV file (e.g., train.csv).
                                Expected columns: 'path', 'emotion'.
            audio_base_dir (str): Base directory where audio files referenced in 'path' are located.
            target_sr (int): Target sample rate to resample audio to.
            max_length_samples (int, optional): Maximum number of samples. Longer files truncated, shorter padded later.
            transforms (callable, optional): Optional transform (e.g., augmentations) applied to the waveform.
            label_map (dict, optional): Dictionary mapping emotion strings to integer labels. If None, uses a default.
        """
        self.audio_base_dir = audio_base_dir
        self.target_sr = target_sr
        self.max_length_samples = max_length_samples
        self.transforms = transforms

        if not os.path.exists(metadata_csv):
            raise FileNotFoundError(f"Metadata CSV file not found: {metadata_csv}")
            
        print(f"Loading metadata from: {metadata_csv}")
        self.metadata = pd.read_csv(metadata_csv)

        # Define or use provided emotion to label mapping
        if label_map is None:
            # Default mapping (adjust as needed for your specific classes)
            self.emotion_map = {
                'neutral': 0, 'calm': 0, 
                'happy': 1,
                'sad': 2,
                'angry': 3,
                'fear': 4, 'fearful': 4,
                'disgust': 5,
                'surprised': 6, 'surprise': 6 
                # Add other mappings if necessary (e.g., 'boredom')
            }
            print("Using default emotion map:", self.emotion_map)
        else:
            self.emotion_map = label_map
            print("Using provided emotion map:", self.emotion_map)

        # Apply mapping and handle potential errors
        self.metadata['label'] = self.metadata['emotion'].map(self.emotion_map)
        
        num_original = len(self.metadata)
        self.metadata = self.metadata.dropna(subset=['label'])
        num_after_drop = len(self.metadata)
        if num_original > num_after_drop:
             print(f"Warning: Dropped {num_original - num_after_drop} rows due to unmappable emotions.")
        
        if not self.metadata.empty:
             self.metadata['label'] = self.metadata['label'].astype(int)
             self.num_classes = self.metadata['label'].max() + 1
             print(f"Dataset contains {self.num_classes} classes after mapping.")
        else:
             print("Warning: Metadata is empty after dropping unmappable emotions.")
             self.num_classes = 0


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx) -> Optional[Tuple[torch.Tensor, int]]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.metadata.iloc[idx]
        relative_audio_path = row['path']
        # Construct full path relative to the project root or a defined base
        # Assumes audio_base_dir points to the root containing dataset folders (e.g., 'data/datasets_raw')
        # and relative_audio_path includes the dataset folder (e.g., 'ravdess/Actor_01/file.wav')
        audio_path = os.path.join(self.audio_base_dir, relative_audio_path)

        try:
            if not os.path.exists(audio_path):
                 raise FileNotFoundError(f"Audio file not found: {audio_path}")

            waveform, sr = torchaudio.load(audio_path)

            # Resample if necessary
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
                waveform = resampler(waveform)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Remove channel dimension (expected by HuBERT) -> [num_samples]
            waveform = waveform.squeeze(0) 

            # Truncate if max_length is set
            if self.max_length_samples is not None and waveform.shape[0] > self.max_length_samples:
                waveform = waveform[:self.max_length_samples]
            
            # Apply augmentations (if any) - applied before padding
            if self.transforms:
                 # Augmentations might expect [batch, time] or [time], ensure compatibility
                 # Assuming transforms take waveform [time] and return augmented [time]
                 waveform = self.transforms(samples=waveform, sample_rate=self.target_sr)
                 # Ensure output is still a tensor
                 if not isinstance(waveform, torch.Tensor):
                      waveform = torch.tensor(waveform, dtype=torch.float32)


            label = row['label']
            
            # Padding will be handled by the collate_fn

            return waveform, torch.tensor(label, dtype=torch.long)

        except FileNotFoundError as e:
            print(f"Error loading file {idx}: {e}")
            return None # Signal error to collate_fn
        except Exception as e:
            print(f"Error processing file {idx} ({audio_path}): {e}")
            return None # Signal error to collate_fn


class AudioDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_names: Union[str, List[str]],
                 data_dir: str, # Base dir containing dataset splits (e.g., data/datasets_processed)
                 audio_dir: str, # Base dir containing raw audio (e.g., data/datasets_raw)
                 batch_size: int = 8,
                 num_workers: int = 4,
                 sample_rate: int = 16000,
                 max_length_seconds: float = 5.0,
                 use_balanced_sampler: bool = False,
                 train_transforms=None,
                 val_transforms=None,
                 test_transforms=None,
                 # feature_type arg removed as we load raw audio now
                 ):
        super().__init__()
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        
        # Store hparams using self.hparams (Lightning convention)
        self.save_hyperparameters()

        # Placeholder datasets and attributes
        self.train_dataset: Optional[Union[Dataset, ConcatDataset]] = None
        self.val_dataset: Optional[Union[Dataset, ConcatDataset]] = None
        self.test_dataset: Optional[Union[Dataset, ConcatDataset]] = None
        self.train_sampler: Optional[WeightedRandomSampler] = None
        self._num_classes: Optional[int] = None
        self._class_weights: Optional[torch.Tensor] = None
        self._label_map: Optional[Dict[str, int]] = None # Store the map used

    def prepare_data(self):
        # Check existence of split files for each dataset
        for ds_name in self.hparams.dataset_names:
            # Splits are expected relative to data_dir
            splits_base = os.path.join(self.hparams.data_dir, ds_name, 'splits') 
            if not os.path.exists(splits_base):
                 raise FileNotFoundError(f"Splits directory not found for {ds_name} at {splits_base}. Run make_split.py.")
            for split in ['train', 'validation', 'test']:
                 split_file = os.path.join(splits_base, f"{split}.csv")
                 if not os.path.exists(split_file):
                     raise FileNotFoundError(f"Split file not found: {split_file}. Run make_split.py.")
        print("Required split CSV files verified.")
        # Audio files existence checked within AudioDataset.__getitem__

    def setup(self, stage: Optional[str] = None):
        train_datasets, val_datasets, test_datasets = [], [], []
        all_train_metadata = []
        
        max_samples = int(self.hparams.max_length_seconds * self.hparams.sample_rate) if self.hparams.max_length_seconds > 0 else None

        # Determine label map from the first dataset (assume consistent across datasets)
        first_ds_splits_dir = os.path.join(self.hparams.data_dir, self.hparams.dataset_names[0], 'splits')
        temp_train_ds = AudioDataset(
            metadata_csv=os.path.join(first_ds_splits_dir, 'train.csv'),
            audio_base_dir=self.hparams.audio_dir, # Base dir for raw audio
            target_sr=self.hparams.sample_rate,
            max_length_samples=max_samples,
            transforms=None # No transforms needed just for setup
        )
        self._label_map = temp_train_ds.emotion_map
        self._num_classes = temp_train_ds.num_classes
        print(f"Determined label map: {self._label_map}")
        print(f"Determined num_classes: {self._num_classes}")


        for ds_name in self.hparams.dataset_names:
            splits_dir = os.path.join(self.hparams.data_dir, ds_name, 'splits')
            # audio_base_dir should point to the root containing the raw audio dataset folders
            # e.g., 'data/datasets_raw' which contains 'ravdess/' and 'crema_d/'
            
            if stage == 'fit' or stage is None:
                train_split_csv = os.path.join(splits_dir, 'train.csv')
                train_ds = AudioDataset(
                    metadata_csv=train_split_csv,
                    audio_base_dir=self.hparams.audio_dir,
                    target_sr=self.hparams.sample_rate,
                    max_length_samples=max_samples,
                    transforms=self.hparams.train_transforms, # Pass train transforms
                    label_map=self._label_map # Use consistent map
                )
                train_datasets.append(train_ds)
                all_train_metadata.append(train_ds.metadata)

                val_split_csv = os.path.join(splits_dir, 'validation.csv')
                val_ds = AudioDataset(
                    metadata_csv=val_split_csv,
                    audio_base_dir=self.hparams.audio_dir,
                    target_sr=self.hparams.sample_rate,
                    max_length_samples=max_samples,
                    transforms=self.hparams.val_transforms, # Pass val transforms (usually None)
                    label_map=self._label_map
                )
                val_datasets.append(val_ds)

            if stage == 'test' or stage is None:
                test_split_csv = os.path.join(splits_dir, 'test.csv')
                test_ds = AudioDataset(
                    metadata_csv=test_split_csv,
                    audio_base_dir=self.hparams.audio_dir,
                    target_sr=self.hparams.sample_rate,
                    max_length_samples=max_samples,
                    transforms=self.hparams.test_transforms, # Pass test transforms (usually None)
                    label_map=self._label_map
                )
                test_datasets.append(test_ds)

        # Combine datasets if multiple were specified
        if len(self.hparams.dataset_names) > 1:
            if train_datasets: self.train_dataset = ConcatDataset(train_datasets)
            if val_datasets: self.val_dataset = ConcatDataset(val_datasets)
            if test_datasets: self.test_dataset = ConcatDataset(test_datasets)
            if all_train_metadata: self.combined_metadata = pd.concat(all_train_metadata, ignore_index=True)
        else: # Only one dataset
            if train_datasets: self.train_dataset = train_datasets[0]
            if val_datasets: self.val_dataset = val_datasets[0]
            if test_datasets: self.test_dataset = test_datasets[0]
            if all_train_metadata: self.combined_metadata = all_train_metadata[0]

        # Calculate weights and sampler for the combined training dataset
        if stage == 'fit' or stage is None:
            if self.train_dataset and self.combined_metadata is not None:
                self._class_weights = self.compute_class_weights()
                if self.hparams.use_balanced_sampler and self._class_weights is not None:
                    print("Calculating sample weights for balanced sampler...")
                    # Need labels for the entire concatenated dataset
                    labels = self.combined_metadata['label'].astype(int).to_numpy()
                    sample_weights = self._class_weights[labels]
                    self.train_sampler = WeightedRandomSampler(
                        weights=sample_weights, 
                        num_samples=len(sample_weights), 
                        replacement=True
                    )
                    print(f"Created WeightedRandomSampler for {len(sample_weights)} samples.")
                else:
                    self.train_sampler = None
            else:
                self._class_weights = None
                self.train_sampler = None

        print(f"Setup complete for stage: {stage}")
        if self.train_dataset: print(f"Train dataset size: {len(self.train_dataset)}")
        if self.val_dataset: print(f"Validation dataset size: {len(self.val_dataset)}")
        if self.test_dataset: print(f"Test dataset size: {len(self.test_dataset)}")

    def compute_class_weights(self) -> Optional[torch.Tensor]:
        """Computes inverse frequency class weights for the combined training dataset."""
        if self.combined_metadata is None or self.combined_metadata.empty:
            print("Warning: Cannot compute class weights, combined metadata not available.")
            return None
        
        print("Calculating class weights...")
        labels = self.combined_metadata['label'].astype(int).tolist()
        if not labels:
             print("Warning: No labels found in combined metadata.")
             return None
             
        class_counts = Counter(labels)
        # Ensure all classes up to max label are present
        for i in range(self.num_classes):
            if i not in class_counts:
                class_counts[i] = 0 # Add missing classes with count 0
                
        # Sort by class index and get counts
        sorted_counts = [class_counts[i] for i in range(self.num_classes)]
        
        # Calculate inverse frequency weights
        total_samples = sum(sorted_counts)
        weights = []
        for count in sorted_counts:
            if count == 0:
                 # Assign a default weight or handle as needed (e.g., weight 1 or 0)
                 # Assigning 0 weight means the loss for this class won't contribute if it appears
                 # Assigning 1 might be safer if it could appear due to augmentation/error
                 weights.append(1.0) 
                 print(f"Warning: Class {len(weights)-1} has 0 samples in training set.")
            else:
                 weights.append(total_samples / (self.num_classes * count))

        class_weights_tensor = torch.tensor(weights, dtype=torch.float32)
        print(f"Computed class weights: {class_weights_tensor.numpy().round(2)}")
        return class_weights_tensor

    @property
    def num_classes(self) -> int:
        if self._num_classes is None:
             # Try to determine from dataset if setup wasn't called properly
             if self.train_dataset:
                 # This might be tricky with ConcatDataset, need metadata
                 if self.combined_metadata is not None:
                      self._num_classes = self.combined_metadata['label'].astype(int).max() + 1
                 else: return 8 # Fallback
             else: return 8 # Fallback
        return self._num_classes

    def train_dataloader(self):
        if self.train_dataset is None:
             raise ValueError("Train dataset not setup. Call setup('fit') first.")
        
        shuffle = self.train_sampler is None # Shuffle if not using balanced sampler
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            sampler=self.train_sampler, # Use the sampler if created
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn # Use custom collate
        )

    def val_dataloader(self):
        if self.val_dataset is None:
             raise ValueError("Validation dataset not setup. Call setup('fit') first.")
        return DataLoader(
            self.val_dataset, 
            batch_size=self.hparams.batch_size, 
            shuffle=False, 
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn # Use custom collate
        )

    def test_dataloader(self):
        if self.test_dataset is None:
             raise ValueError("Test dataset not setup. Call setup('test') first.")
        return DataLoader(
            self.test_dataset, 
            batch_size=self.hparams.batch_size, 
            shuffle=False, 
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn # Use custom collate
        )
        
    def collate_fn(self, batch: List[Optional[Tuple[torch.Tensor, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        Processes a batch of samples:
        1. Filters out None items (errors during loading).
        2. Pads waveforms to the max length in the batch.
        3. Creates an attention mask.
        4. Stacks labels.
        Returns a dictionary compatible with Hugging Face model's forward pass.
        """
        # Filter out None samples
        batch = [item for item in batch if item is not None]
        if not batch:
            # Return empty tensors or dict with empty tensors
            return {'input_values': torch.empty(0), 'attention_mask': torch.empty(0), 'labels': torch.empty(0)}

        waveforms = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        # Pad waveforms
        waveforms_padded = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True, padding_value=0.0)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.zeros_like(waveforms_padded, dtype=torch.long)
        for i, wf in enumerate(waveforms):
            attention_mask[i, :len(wf)] = 1
            
        # Stack labels
        labels_stacked = torch.stack(labels)
        
        return {
            'input_values': waveforms_padded,
            'attention_mask': attention_mask,
            'labels': labels_stacked
        }

# Example Usage (for testing)
if __name__ == '__main__':
    # Assumes you have 'data/datasets_processed' with splits and 'data/datasets_raw' with audio
    processed_data_dir = '../../data/datasets_processed' # Adjust path relative to this file
    raw_audio_dir = '../../data/datasets_raw'       # Adjust path relative to this file
    
    # Check if directories exist before running
    if not os.path.isdir(processed_data_dir) or not os.path.isdir(raw_audio_dir):
         print(f"Skipping example: Ensure processed data dir ({processed_data_dir}) and raw audio dir ({raw_audio_dir}) exist.")
    else:
        print("\nTesting Combined (RAVDESS + CREMA-D) AudioDataModule...")
        
        # Placeholder for augmentation function if needed
        # from augment import build_augmentations
        # train_transforms = build_augmentations(sample_rate=16000)
        train_transforms = None 

        combined_dm = AudioDataModule(
            dataset_names=['ravdess', 'crema_d'],
            data_dir=processed_data_dir, # Dir with splits CSVs
            audio_dir=raw_audio_dir,     # Dir with raw audio folders
            batch_size=4,
            num_workers=0, # Use 0 workers for easier debugging
            sample_rate=16000,
            max_length_seconds=4.0,
            use_balanced_sampler=True, # Test balanced sampler
            train_transforms=train_transforms
        )
        try:
            combined_dm.prepare_data()
            combined_dm.setup('fit')
            
            print(f"Num classes found: {combined_dm.num_classes}")
            print(f"Class weights: {combined_dm.compute_class_weights()}")

            train_loader = combined_dm.train_dataloader()
            print(f"Combined train loader length: {len(train_loader)}")
            
            # Check a few batches
            for i, batch in enumerate(train_loader):
                if i >= 2: break # Check first 2 batches
                print(f"\nTrain Batch {i+1} shapes:")
                print("Input Values:", batch['input_values'].shape)
                print("Attention Mask:", batch['attention_mask'].shape)
                print("Labels:", batch['labels'].shape)
                # Verify mask matches padding
                assert batch['input_values'].shape == batch['attention_mask'].shape
                # Check a sample mask
                print("Sample 0 Mask Sum:", batch['attention_mask'][0].sum())
                print("Sample 0 Waveform Length (estimated from mask):", batch['attention_mask'][0].sum().item())


            # Test validation loader
            combined_dm.setup('validate') # Ensure val dataset is loaded
            val_loader = combined_dm.val_dataloader()
            print(f"\nCombined val loader length: {len(val_loader)}")
            batch = next(iter(val_loader))
            print("Validation Batch shapes:")
            print("Input Values:", batch['input_values'].shape)
            print("Attention Mask:", batch['attention_mask'].shape)
            print("Labels:", batch['labels'].shape)

        except FileNotFoundError as e:
            print(f"\nSkipping Combined test due to missing file/dir: {e}")
        except Exception as e:
            print(f"\nError during Combined test: {e}")
            import traceback
            traceback.print_exc()
