import pickle
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TedHumorDataset(Dataset):
    """
    PyTorch Dataset class for loading pre-extracted TED-Humor dataset features.
    Loads Covarep audio features and humor labels from pickle files.
    """
    def __init__(self, dataset_path, split='train'):
        """
        Args:
            dataset_path (str): Path to the directory containing the pickle files
                                (e.g., "datasets/humor_datasets/ted_humor_sdk_v1/final_humor_sdk").
            split (str): Dataset split ('train', 'dev', 'test').
        """
        logging.info(f"Initializing TedHumorDataset for split '{split}' from: {dataset_path}")

        self.dataset_path = dataset_path
        self.split = split

        # Define paths to pickle files
        self.data_folds_path = os.path.join(dataset_path, "data_folds.pkl")
        self.humor_labels_path = os.path.join(dataset_path, "humor_label_sdk.pkl")
        self.audio_features_path = os.path.join(dataset_path, "covarep_features_sdk.pkl")

        # Load data
        try:
            with open(self.data_folds_path, 'rb') as f:
                self.data_folds = pickle.load(f)
            with open(self.humor_labels_path, 'rb') as f:
                self.humor_labels = pickle.load(f)
            with open(self.audio_features_path, 'rb') as f:
                self.audio_features = pickle.load(f)
        except FileNotFoundError as e:
            logging.error(f"Required pickle file not found: {e}")
            raise
        except Exception as e:
            logging.error(f"Error loading pickle files: {e}")
            raise

        # Get indices for the specified split
        if self.split not in self.data_folds:
            raise ValueError(f"Invalid split: '{self.split}'. Available splits: {list(self.data_folds.keys())}")

        self.indices = self.data_folds[self.split]
        logging.info(f"Loaded {len(self.indices)} samples for split '{self.split}'.")

        # Basic validation: Check if indices are valid for labels and features
        if not all(idx in self.humor_labels for idx in self.indices):
             logging.warning(f"Some indices in '{self.split}' split are not found in humor_label_sdk.pkl")
        if not all(idx in self.audio_features for idx in self.indices):
             logging.warning(f"Some indices in '{self.split}' split are not found in covarep_features_sdk.pkl")


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """Loads a single sample (audio features and humor label)."""
        sample_id = self.indices[idx]

        try:
            # Get audio features (Covarep)
            # Assuming features are numpy arrays
            audio_features = self.audio_features[sample_id]
            # Convert numpy array to torch tensor
            audio_features_tensor = torch.tensor(audio_features, dtype=torch.float)

            # Get humor label
            # Assuming labels are scalar (0 or 1)
            humor_label = self.humor_labels[sample_id]
            humor_label_tensor = torch.tensor(humor_label, dtype=torch.long) # Use long for classification

            # For compatibility with existing models expecting attention mask, create a dummy one
            # The shape should match the sequence length of the features
            audio_attention_mask = torch.ones(audio_features_tensor.shape[0], dtype=torch.long)

            # Return sample dictionary
            sample = {
                'audio_input_values': audio_features_tensor,
                'audio_attention_mask': audio_attention_mask, # Dummy mask
                'label': humor_label_tensor, # Primary label for binary classification
                'source': 'ted_humor',
                'has_video': torch.tensor(False, dtype=torch.bool), # No video for this dataset
                # Include other potential labels as None for compatibility if needed by model
                'emotion_label': None,
                'smile_label': None,
                'joke_label': None,
                'text_input_ids': None, # No text for this dataset
                'text_attention_mask': None # No text for this dataset
            }

            return sample

        except KeyError:
            logging.error(f"Sample ID {sample_id} not found in all pickle files. Skipping sample.")
            return None # Signal to skip this sample
        except Exception as e:
            logging.error(f"Error processing sample {sample_id}: {e}", exc_info=True)
            return None # Signal to skip this sample


# Custom collate function to handle None samples (skipped samples)
def collate_fn_skip_none(batch):
    """
    Filters out None samples from the batch and stacks the rest.
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None # Return None if the entire batch was skipped

    # Assuming all remaining samples have the same keys and compatible shapes
    # This is a simplified collate. For variable length sequences, padding is needed.
    # Since Covarep features might have variable length, we need padding here.

    # Find max sequence length for padding
    max_audio_len = max(item['audio_input_values'].shape[0] for item in batch)

    # Pad audio features and attention masks
    padded_batch = {}
    for key in batch[0].keys():
        if key in ['audio_input_values', 'audio_attention_mask']:
            # Pad sequences
            padded_sequences = []
            for item in batch:
                seq = item[key]
                pad_len = max_audio_len - seq.shape[0]
                # Pad with 0s for features, 0s for attention mask
                padded_seq = torch.nn.functional.pad(seq, (0, pad_len), "constant", 0)
                padded_sequences.append(padded_seq)
            padded_batch[key] = torch.stack(padded_sequences)
        elif key in ['label', 'has_video']:
            # Stack tensors directly
            padded_batch[key] = torch.stack([item[key] for item in batch])
        elif key in ['text_input_ids', 'text_attention_mask', 'emotion_label', 'smile_label', 'joke_label']:
             # Handle None or stack if not None (assuming they are None for this dataset)
             # If they were not None, we'd need padding logic similar to audio
             if batch[0][key] is None:
                  padded_batch[key] = None # Keep as None if all are None
             else:
                  # This case shouldn't happen for this specific dataset based on inspection,
                  # but adding a placeholder for robustness.
                  try:
                       padded_batch[key] = torch.stack([item[key] for item in batch])
                  except Exception as e:
                       logging.warning(f"Could not stack items for key '{key}': {e}. Returning None for this key.")
                       padded_batch[key] = None
        else: # Handle other keys like 'source'
            padded_batch[key] = [item[key] for item in batch] # Keep as list

    return padded_batch


# --- Example Usage (for testing this script directly) ---
if __name__ == '__main__':
    print("--- Testing TedHumorDataset ---")

    # Assuming the zip was extracted to datasets/humor_datasets/ted_humor_sdk_v1
    DATASET_ROOT = "datasets/humor_datasets/ted_humor_sdk_v1/final_humor_sdk"

    try:
        # Test train split
        dataset_train = TedHumorDataset(
            dataset_path=DATASET_ROOT,
            split='train'
        )
        print(f"\nTrain Dataset size: {len(dataset_train)}")

        if len(dataset_train) > 0:
            print("\nLoading train sample 0...")
            sample0 = dataset_train[0]
            if sample0: # Check if sample loading succeeded
                print(" Sample 0 keys:", sample0.keys())
                print(" Audio Features shape:", sample0['audio_input_values'].shape)
                print(" Audio Attention Mask shape:", sample0['audio_attention_mask'].shape)
                print(" Label:", sample0.get('label'))
                print(" Source:", sample0['source'])
                print(" Has Video:", sample0['has_video'])
                print(" Emotion Label:", sample0.get('emotion_label')) # Should be None
                print(" Text Input IDs:", sample0.get('text_input_ids')) # Should be None
            else:
                print(" Sample 0 failed to load.")

        # Test validation split
        dataset_val = TedHumorDataset(
            dataset_path=DATASET_ROOT,
            split='dev' # Use 'dev' as validation split based on inspection
        )
        print(f"\nValidation Dataset size: {len(dataset_val)}")
        if len(dataset_val) > 0:
            print("\nLoading validation sample 0...")
            sample_val = dataset_val[0]
            if sample_val:
                print(" Val Sample keys:", sample_val.keys())
                print(" Audio Features shape:", sample_val['audio_input_values'].shape)
                print(" Label:", sample_val.get('label'))
            else:
                print(" Validation sample 0 failed to load.")

        # Test test split
        dataset_test = TedHumorDataset(
            dataset_path=DATASET_ROOT,
            split='test'
        )
        print(f"\nTest Dataset size: {len(dataset_test)}")
        if len(dataset_test) > 0:
            print("\nLoading test sample 0...")
            sample_test = dataset_test[0]
            if sample_test:
                print(" Test Sample keys:", sample_test.keys())
                print(" Audio Features shape:", sample_test['audio_input_values'].shape)
                print(" Label:", sample_test.get('label'))
            else:
                print(" Test sample 0 failed to load.")


    except Exception as e:
        print(f"\nError during dataset test: {e}")
        logging.error("Dataset test failed", exc_info=True)

    print("\n--- TedHumorDataset Test Finished ---")
