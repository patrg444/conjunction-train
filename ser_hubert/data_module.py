import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict, Audio, Features, Value
import torch
import torchaudio
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any # Added Any
from transformers import AutoFeatureExtractor
import torch.nn.functional as F # Added for padding if needed later

# Define your label mapping (adjust as needed)
LABEL_MAP = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    # Add other labels if necessary (e.g., surprise, calm from RAVDESS/CREMA-D)
    # Ensure this matches the num_classes in HubertSER module
    "unknown": -1 # Example for handling labels not in the map
}


class AudioDatasetBase(Dataset):
    """Base class for audio datasets handling common loading and processing."""
    def __init__(self, feature_extractor, max_duration_s=5.0, label_map=None, label_col="emotion", data_root=None):
        self.feature_extractor = feature_extractor
        self.max_duration_s = max_duration_s
        self.target_sr = feature_extractor.sampling_rate if feature_extractor else 16000
        self.label_map = label_map if label_map is not None else {}
        self.label_col = label_col
        self.data_root = data_root # Optional root dir for relative paths

    def _load_and_process_audio(self, filepath):
        # Resolve path if data_root is provided
        if self.data_root and not os.path.isabs(filepath):
             # Simple join, assumes filepath is relative to data_root
             filepath = os.path.join(self.data_root, filepath)
        # Else: Assume filepath is absolute or relative to CWD

        try:
            wav_tensor, sr = torchaudio.load(filepath)
            if wav_tensor.shape[0] > 1: # Ensure mono
                wav_tensor = torch.mean(wav_tensor, dim=0, keepdim=True)
        except Exception as e:
            print(f"Error loading audio file {filepath}: {e}")
            raise RuntimeError(f"Failed to load audio: {filepath}") from e

        # Resample if necessary
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            wav_tensor = resampler(wav_tensor)
            sr = self.target_sr

        # Truncate/Pad
        max_samples = int(self.max_duration_s * sr)
        current_samples = wav_tensor.shape[1]
        if current_samples > max_samples:
            wav_tensor = wav_tensor[:, :max_samples]
        # Optional padding:
        # elif current_samples < max_samples:
        #     padding_needed = max_samples - current_samples
        #     wav_tensor = F.pad(wav_tensor, (0, padding_needed), "constant", 0)


        # Squeeze channel dim for feature extractor
        wav_tensor_proc = wav_tensor.squeeze(0)

        # Extract features
        features = self.feature_extractor(wav_tensor_proc.to(torch.float32), sampling_rate=sr, return_tensors="pt")
        input_values = features.input_values.squeeze(0) # Remove batch dim
        attention_mask = features.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(0)

        return input_values, attention_mask

    def _get_label_id(self, item):
        # item can be a dict (from HF dataset) or a Series (from DataFrame)
        label_str = item.get(self.label_col, "unknown") # Default to unknown if column missing
        if pd.isna(label_str): # Handle potential NaN/None
            label_str = "unknown"
        label_id = self.label_map.get(str(label_str).lower(), self.label_map.get("unknown", -1))
        return torch.tensor(label_id, dtype=torch.long)


class SplitAudioDataset(AudioDatasetBase):
    """Dataset wrapper for Hugging Face datasets loaded from train/val/test splits."""
    def __init__(self, hf_dataset, feature_extractor, augment_fn=None, max_duration_s=5.0, label_map=None, label_col="emotion", data_root=None):
        super().__init__(feature_extractor, max_duration_s, label_map, label_col, data_root)
        self.hf_dataset = hf_dataset
        self.augment_fn = augment_fn # Noise augmentation factor (std dev)

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        filepath = item["filepath"] # Path comes from the split CSV

        input_values, attention_mask = self._load_and_process_audio(filepath)

        # Apply noise augmentation
        if self.augment_fn and self.augment_fn > 0:
            input_values = input_values.to(torch.float32)
            noise = torch.randn_like(input_values) * self.augment_fn
            input_values = input_values + noise

        label_id = self._get_label_id(item)

        output = {"input_values": input_values, "labels": label_id}
        if attention_mask is not None:
            output["attention_mask"] = attention_mask
        return output


class ManifestAudioDataset(AudioDatasetBase):
    """Dataset wrapper for datasets loaded from a single manifest CSV file."""
    def __init__(self, manifest_df, feature_extractor, max_duration_s=5.0, label_map=None, label_col="emotion", path_col="path", data_root=None):
        super().__init__(feature_extractor, max_duration_s, label_map, label_col, data_root)
        self.manifest_df = manifest_df
        self.path_col = path_col # Allow specifying the path column name

    def __len__(self):
        return len(self.manifest_df)

    def __getitem__(self, idx):
        item = self.manifest_df.iloc[idx] # item is now a pandas Series
        filepath = item[self.path_col] # Path comes from the manifest CSV

        input_values, attention_mask = self._load_and_process_audio(filepath)
        label_id = self._get_label_id(item)

        output = {"input_values": input_values, "labels": label_id}
        if attention_mask is not None:
            output["attention_mask"] = attention_mask
        return output


@dataclass
class DataCollatorAudio: # Renamed for clarity
    """
    Data collator that will dynamically pad the audio inputs received.
    Args:
        feature_extractor (:class:`~transformers.AutoFeatureExtractor`)
            The feature_extractor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    feature_extractor: AutoFeatureExtractor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None


    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features] # Treat labels like input_ids for padding

        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Keep labels as a simple tensor list, padding is not usually needed for sequence classification labels
        # However, if a collator expects padding for labels too (less common), handle it here.
        # For simple classification, just stack the labels.
        batch["labels"] = torch.stack([feature["labels"] for feature in features])

        return batch


class CREMADRAVDESSDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str = "splits", # Directory containing train.csv, val.csv, test.csv
                 batch_size: int = 8,
                 num_workers: int = 4,
                 model_name: str = "facebook/hubert-base-ls960", # To get feature extractor
                 max_duration_s: float = 5.0,
                  label_map: dict = None,
                  add_noise: bool = True, # Control augmentation
                  noise_std: float = 0.01, # Noise level (standard deviation)
                  data_root: str = None): # Added data_root parameter
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_name = model_name
        self.max_duration_s = max_duration_s
        self.label_map = label_map if label_map is not None else LABEL_MAP
        self.data_root = data_root # Store data_root
        self.feature_extractor = None
        self.dataset_dict = None
        self.data_collator = None # Added collator instance
        self.add_noise = add_noise
        self.noise_std = noise_std

        # Define augmentation scale (standard deviation)
        if self.add_noise:
             self.train_augment = self.noise_std # Use noise_std directly
        else:
             self.train_augment = None

    def setup(self, stage: str | None = None):
        # Moved import to top level
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
        target_sr = self.feature_extractor.sampling_rate

        # Instantiate the data collator
        self.data_collator = DataCollatorAudio(feature_extractor=self.feature_extractor, padding=True) # Use renamed collator


        # Load datasets from CSV files
        self.dataset_dict = load_dataset(
            "csv",
            data_files={
                "train": os.path.join(self.data_dir, "train.csv"),
                "val": os.path.join(self.data_dir, "val.csv"),
                "test": os.path.join(self.data_dir, "test.csv"),
            },
            # Define features based *only* on the CSV columns
            features=Features({
                "filepath": Value("string"), # Changed from "path"
                "laugh": Value("int64"),     # Changed from "emotion", assuming 0/1 int labels
                "split": Value("string")     # Added "split" column
            })
        )

        # Create wrapped datasets using SplitAudioDataset
        # Pass data_root if paths in CSV are relative to it (assuming they are relative to CWD for now)
        self.train_dataset = SplitAudioDataset(self.dataset_dict["train"], self.feature_extractor, augment_fn=self.train_augment, max_duration_s=self.max_duration_s, label_map=self.label_map)
        self.val_dataset = SplitAudioDataset(self.dataset_dict["val"], self.feature_extractor, max_duration_s=self.max_duration_s, label_map=self.label_map)
        self.test_dataset = SplitAudioDataset(self.dataset_dict["test"], self.feature_extractor, max_duration_s=self.max_duration_s, label_map=self.label_map)

    # --- DataLoaders ---
    # Note: Implement custom collate_fn if needed, especially for dynamic padding
    # from transformers import Wav2Vec2FeatureExtractor # Or HubertFeatureExtractor etc.
    # def collate_fn(batch):
    #    # Padding logic here using self.feature_extractor.pad
    #    pass

    def train_dataloader(self):
        # Add balanced sampler here if doing cross-corpus training
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=self.data_collator)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.data_collator)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.data_collator)


class ManifestDataModule(pl.LightningDataModule):
    """DataModule for loading data from a single manifest CSV file."""
    def __init__(self,
                 manifest_path: str, # Path to the manifest CSV
                 batch_size: int = 8,
                 num_workers: int = 4,
                 model_name: str = "facebook/hubert-base-ls960",
                 max_duration_s: float = 5.0,
                 label_map: dict = None,
                 label_col: str = "emotion", # Column name for labels in CSV
                 path_col: str = "path",     # Column name for file paths in CSV
                 data_root: str = None):     # Optional root directory for relative paths
        super().__init__()
        self.manifest_path = manifest_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_name = model_name
        self.max_duration_s = max_duration_s
        self.label_map = label_map if label_map is not None else LABEL_MAP
        self.label_col = label_col
        self.path_col = path_col
        self.data_root = data_root
        self.feature_extractor = None
        self.manifest_dataset = None
        self.data_collator = None

    def setup(self, stage: str | None = None):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
        self.data_collator = DataCollatorAudio(feature_extractor=self.feature_extractor, padding=True)

        # Load the manifest CSV into a pandas DataFrame
        if not os.path.exists(self.manifest_path):
             raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")
        manifest_df = pd.read_csv(self.manifest_path)

        # Ensure required columns exist
        if self.path_col not in manifest_df.columns:
            raise ValueError(f"Path column '{self.path_col}' not found in manifest: {self.manifest_path}")
        if self.label_col not in manifest_df.columns:
             raise ValueError(f"Label column '{self.label_col}' not found in manifest: {self.manifest_path}")


        # Create the dataset using ManifestAudioDataset
        self.manifest_dataset = ManifestAudioDataset(
            manifest_df=manifest_df,
            feature_extractor=self.feature_extractor,
            max_duration_s=self.max_duration_s,
            label_map=self.label_map,
            label_col=self.label_col,
            path_col=self.path_col,
            data_root=self.data_root # Pass data_root if paths are relative
        )

    def manifest_dataloader(self):
        """Returns the DataLoader for the manifest dataset."""
        if self.manifest_dataset is None:
            raise RuntimeError("The setup() method must be called before manifest_dataloader().")
        # Shuffle=False is typical for inference/embedding generation
        return DataLoader(
            self.manifest_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
            shuffle=False
        )

    # Provide dummy train/val/test dataloaders returning None or raising error
    # if this module is used in a context expecting them (like Pytorch Lightning Trainer)
    def train_dataloader(self):
        # return None # Or raise error
        raise NotImplementedError("ManifestDataModule does not provide a train_dataloader.")

    def val_dataloader(self):
        # return None
        raise NotImplementedError("ManifestDataModule does not provide a val_dataloader.")

    def test_dataloader(self):
        # return None
        raise NotImplementedError("ManifestDataModule does not provide a test_dataloader.")


if __name__ == '__main__':
    # Example usage for CREMADRAVDESSDataModule (requires CSV files in ./splits/)
    # Create dummy CSVs for testing if needed
    # os.makedirs("splits", exist_ok=True)
    # pd.DataFrame({'path': ['path/to/dummy1.wav'], 'speaker': ['1'], 'emotion': ['happy']}).to_csv("splits/train.csv", index=False)
    # pd.DataFrame({'path': ['path/to/dummy2.wav'], 'speaker': ['2'], 'emotion': ['sad']}).to_csv("splits/val.csv", index=False)
    # pd.DataFrame({'path': ['path/to/dummy3.wav'], 'speaker': ['3'], 'emotion': ['angry']}).to_csv("splits/test.csv", index=False)
    # print("Dummy CSVs created in ./splits/ (if they didn't exist)")

    print("Initializing DataModule...")
    # Ensure you have a feature extractor cache or are online
    dm = CREMADRAVDESSDataModule(data_dir="splits")
    print("Setting up DataModule...")
    dm.setup()
    print("DataModule setup complete.")

    print("\nChecking train_dataloader...")
    try:
        train_loader = dm.train_dataloader()
        for i, batch in enumerate(train_loader):
            print("Train Batch {}:".format(i+1)) # Line 197 (Replaced f-string)
            print("  Input values shape:", batch['input_values'].shape)
            if 'attention_mask' in batch:
                print("  Attention mask shape:", batch['attention_mask'].shape)
            print("  Labels:", batch['labels'])
            if i >= 1: # Check first 2 batches
                break
        print("Train DataLoader check passed.")
    except Exception as e:
        print(f"Error checking train_dataloader: {e}")
        import traceback
        traceback.print_exc()
