import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class HumorMultimodalDatasetV2(Dataset):
    def __init__(self, manifest_path, id_col='talk_id', label_col='label', 
                 audio_col='sequential_audio_path', text_col='pooled_full_text_path', 
                 video_au_col='sequential_video_au_path',
                 max_audio_len=None, max_video_au_len=None,
                 audio_pad_value=0.0, video_au_pad_value=0.0):
        """
        Args:
            manifest_path (str): Path to the manifest CSV file.
            id_col (str): Column name for clip IDs.
            label_col (str): Column name for labels.
            audio_col (str): Column name for sequential audio embedding paths.
            text_col (str): Column name for pooled text embedding paths.
            video_au_col (str): Column name for sequential video AU embedding paths.
            max_audio_len (int, optional): Maximum length for audio sequences. Truncates or pads.
            max_video_au_len (int, optional): Maximum length for video AU sequences. Truncates or pads.
            audio_pad_value (float): Value to use for padding audio sequences.
            video_au_pad_value (float): Value to use for padding video AU sequences.
        """
        logger.info(f"Loading manifest from {manifest_path}")
        self.manifest_df = pd.read_csv(manifest_path)
        self.id_col = id_col
        self.label_col = label_col
        self.audio_col = audio_col
        self.text_col = text_col
        self.video_au_col = video_au_col
        self.max_audio_len = max_audio_len
        self.max_video_au_len = max_video_au_len
        self.audio_pad_value = audio_pad_value
        self.video_au_pad_value = video_au_pad_value

        # Pre-filter entries with missing essential (audio/text) embeddings if paths are NaN
        self.manifest_df.dropna(subset=[self.audio_col, self.text_col], inplace=True)
        self.manifest_df = self.manifest_df[self.manifest_df[self.audio_col] != 'None']
        self.manifest_df = self.manifest_df[self.manifest_df[self.text_col] != 'None']
        self.manifest_df.reset_index(drop=True, inplace=True)
        logger.info(f"Manifest loaded. Number of samples after filtering missing audio/text: {len(self.manifest_df)}")


    def __len__(self):
        return len(self.manifest_df)

    def _load_npy(self, path, feature_name="feature"):
        if pd.isna(path) or path == 'None' or not os.path.exists(path):
            logger.debug(f"{feature_name} path is invalid or file does not exist: {path}")
            return None
        try:
            return np.load(path)
        except Exception as e:
            logger.error(f"Error loading {feature_name} from {path}: {e}", exc_info=True)
            return None

    def _pad_or_truncate(self, array, max_len, pad_value):
        if array is None: # Should ideally not happen if pre-filtered, but as a safeguard
            return np.full((max_len, 1) if max_len else (0,1) , pad_value, dtype=np.float32) # Assuming 1D feature if dim unknown
            
        current_len = array.shape[0]
        feature_dim = array.shape[1] if array.ndim > 1 else 1
        
        if array.ndim == 1 and feature_dim == 1 and current_len > 0 : # If 1D array, reshape to (current_len, 1)
             if array.shape[0] == feature_dim and current_len == 1: # special case for (D,) shaped embeddings
                pass # keep as is
             else:
                array = array.reshape(-1,1)


        if max_len is None:
            return array

        if current_len > max_len:
            return array[:max_len, :] if array.ndim > 1 else array[:max_len]
        elif current_len < max_len:
            padding_shape = (max_len - current_len, feature_dim) if array.ndim > 1 else (max_len - current_len,)
            padding = np.full(padding_shape, pad_value, dtype=array.dtype)
            return np.concatenate((array, padding), axis=0)
        return array

    def __getitem__(self, idx):
        row = self.manifest_df.iloc[idx]
        
        audio_path = row[self.audio_col]
        text_path = row[self.text_col]
        video_au_path = row[self.video_au_col]
        
        label = int(row[self.label_col])
        
        # Load audio features
        audio_features = self._load_npy(audio_path, "Audio")
        if audio_features is None: # Should have been filtered by dropna, but as a fallback
            audio_features = np.array([[self.audio_pad_value]], dtype=np.float32) # Placeholder
            logger.warning(f"Audio features None for idx {idx}, path {audio_path}. Using placeholder.")
        if self.max_audio_len:
            audio_features = self._pad_or_truncate(audio_features, self.max_audio_len, self.audio_pad_value)

        # Load text features (already pooled, so no truncation/padding needed here unless specified differently)
        text_features = self._load_npy(text_path, "Text")
        if text_features is None: # Should have been filtered
            text_features = np.array([0.0], dtype=np.float32) # Placeholder, ensure correct dim later
            logger.warning(f"Text features None for idx {idx}, path {text_path}. Using placeholder.")


        # Load video AU features
        video_au_features = self._load_npy(video_au_path, "Video AU")
        if video_au_features is None or video_au_features.size == 0: # Handles completely missing or empty .npy
            # Create a zero array of shape (max_video_au_len, num_aus) or (0, num_aus) if max_len is None
            # Assuming num_aus can be inferred or is fixed (e.g. 17 for AU_INTENSITY_COLUMNS)
            num_aus = len(AU_INTENSITY_COLUMNS) 
            target_shape = (self.max_video_au_len if self.max_video_au_len else 0, num_aus)
            video_au_features = np.full(target_shape, self.video_au_pad_value, dtype=np.float32)
            if video_au_path and os.path.exists(video_au_path) and self._load_npy(video_au_path).size == 0 :
                 logger.debug(f"Video AU data at {video_au_path} is empty. Using zeros of shape {target_shape}.")
            else:
                 logger.debug(f"Video AU data not found or invalid for {video_au_path}. Using zeros of shape {target_shape}.")

        if self.max_video_au_len:
             # Ensure video_au_features has 2 dims before padding, even if it's an empty placeholder
            if video_au_features.ndim == 1 and video_au_features.shape[0] == 0 : # (0,)
                video_au_features = video_au_features.reshape(0, len(AU_INTENSITY_COLUMNS)) # (0, num_aus)
            elif video_au_features.ndim == 1 and video_au_features.shape[0] > 0: # (N,) should not happen for AUs
                logger.warning(f"Unexpected 1D video AU features for {row[self.id_col]}, attempting reshape.")
                video_au_features = video_au_features.reshape(-1, len(AU_INTENSITY_COLUMNS))


            video_au_features = self._pad_or_truncate(video_au_features, self.max_video_au_len, self.video_au_pad_value)


        return {
            'audio_features': torch.FloatTensor(audio_features),
            'text_features': torch.FloatTensor(text_features),
            'video_au_features': torch.FloatTensor(video_au_features),
            'label': torch.tensor(label, dtype=torch.long)
        }

if __name__ == '__main__':
    # Example Usage (for testing the dataset class)
    # Create dummy manifest and embedding files to test
    
    # Base directory for dummy data
    dummy_base_dir = "dummy_humor_data"
    dummy_manifest_path = os.path.join(dummy_base_dir, "dummy_manifest.csv")
    dummy_audio_emb_dir = os.path.join(dummy_base_dir, "audio_sequential")
    dummy_text_emb_dir = os.path.join(dummy_base_dir, "text_pooled_full")
    dummy_video_au_emb_dir = os.path.join(dummy_base_dir, "video_sequential_au")

    os.makedirs(dummy_audio_emb_dir, exist_ok=True)
    os.makedirs(dummy_text_emb_dir, exist_ok=True)
    os.makedirs(dummy_video_au_emb_dir, exist_ok=True)

    sample_data = []
    num_samples = 5
    audio_feature_dim = 1024 # Example WavLM dim
    text_feature_dim = 1024  # Example XLM-R Large dim
    video_au_dim = len(AU_INTENSITY_COLUMNS)

    for i in range(num_samples):
        talk_id = f"sample_{i+1}"
        label = i % 2
        split = 'train' if i < 3 else 'val'
        
        # Create dummy audio embedding
        audio_len = np.random.randint(50, 150)
        audio_emb = np.random.rand(audio_len, audio_feature_dim).astype(np.float32)
        audio_path = os.path.join(dummy_audio_emb_dir, f"{talk_id}.npy")
        np.save(audio_path, audio_emb)
        
        # Create dummy text embedding
        text_emb = np.random.rand(text_feature_dim).astype(np.float32)
        text_path = os.path.join(dummy_text_emb_dir, f"{talk_id}.npy")
        np.save(text_path, text_emb)
        
        # Create dummy video AU embedding (some might be empty)
        video_path = os.path.join(dummy_video_au_emb_dir, f"{talk_id}.npy")
        if i % 3 != 0: # Make some video AUs "missing" or empty
            video_au_len = np.random.randint(30, 100)
            video_au_emb = np.random.rand(video_au_len, video_au_dim).astype(np.float32)
            np.save(video_path, video_au_emb)
        else: # Create an empty file for one case, and no file for another
            if i == 0: # empty file
                 np.save(video_path, np.array([], dtype=np.float32))
            # else: for i=3, no file will be created, simulating missing OpenFace output
                 
        sample_data.append({
            'talk_id': talk_id, 'label': label, 'split': split,
            'sequential_audio_path': audio_path,
            'pooled_full_text_path': text_path,
            'sequential_video_au_path': video_path if os.path.exists(video_path) else None # Path if exists
        })

    # Create one sample with missing text to test filtering
    talk_id_missing = "sample_missing_text"
    audio_path_missing = os.path.join(dummy_audio_emb_dir, f"{talk_id_missing}.npy")
    np.save(audio_path_missing, np.random.rand(100, audio_feature_dim).astype(np.float32))
    sample_data.append({
        'talk_id': talk_id_missing, 'label': 0, 'split': 'train',
        'sequential_audio_path': audio_path_missing,
        'pooled_full_text_path': None, # Missing text
        'sequential_video_au_path': None
    })


    dummy_manifest_df = pd.DataFrame(sample_data)
    dummy_manifest_df.to_csv(dummy_manifest_path, index=False)
    
    logger.info(f"Dummy manifest created at {dummy_manifest_path}")

    # Test the Dataset
    dataset = HumorMultimodalDatasetV2(
        manifest_path=dummy_manifest_path,
        max_audio_len=100,
        max_video_au_len=80
    )
    
    logger.info(f"Dataset size: {len(dataset)}")
    if len(dataset) > 0:
        for i in range(min(len(dataset), 5)):
            sample = dataset[i]
            logger.info(f"Sample {i}:")
            logger.info(f"  Audio shape: {sample['audio_features'].shape}")
            logger.info(f"  Text shape: {sample['text_features'].shape}")
            logger.info(f"  Video AU shape: {sample['video_au_features'].shape}")
            logger.info(f"  Label: {sample['label']}")
    
    # Clean up dummy files and directory
    # import shutil
    # shutil.rmtree(dummy_base_dir)
    # logger.info(f"Cleaned up dummy data directory: {dummy_base_dir}")
