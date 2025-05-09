import pandas as pd
import pickle
import argparse
import os
import random
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def create_full_dataset_manifest(pickle_label_path, embeddings_manifest_path, output_manifest_path,
                                 train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Creates a comprehensive multimodal manifest by:
    1. Loading talk_ids and labels from the UR-FUNNY humor_label_sdk.pkl file.
    2. Generating train/val/test splits for these talk_ids.
    3. Merging this information with an existing manifest that contains embedding paths.

    Args:
        pickle_label_path (str): Path to the humor_label_sdk.pkl file.
        embeddings_manifest_path (str): Path to the CSV manifest containing 'talk_id' and
                                        paths to text, audio, and video embeddings.
        output_manifest_path (str): Path to save the final comprehensive CSV manifest.
        train_ratio (float): Proportion of data for the training set.
        val_ratio (float): Proportion of data for the validation set.
        test_ratio (float): Proportion of data for the test set.
        random_seed (int): Random seed for reproducible splits.
    """
    if not abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-9: # Check for sum to 1.0 with tolerance
        logger.error(f"Train ({train_ratio}), validation ({val_ratio}), and test ({test_ratio}) ratios must sum to 1.0. Current sum: {train_ratio + val_ratio + test_ratio}")
        return

    logger.info(f"Loading labels from pickle: {pickle_label_path}")
    try:
        with open(pickle_label_path, 'rb') as f:
            label_data_sdk = pickle.load(f)
        if not isinstance(label_data_sdk, dict):
            logger.error(f"Pickle file {pickle_label_path} did not contain a dictionary.")
            return
        
        # Convert SDK data (keys are talk_ids, values are labels) to DataFrame
        # Keys from humor_label_sdk.pkl are integers, convert to string for talk_id
        label_df_data = [{'talk_id': str(k), 'label': int(v)} for k, v in label_data_sdk.items()]
        label_df = pd.DataFrame(label_df_data)
        logger.info(f"Loaded {len(label_df)} talk_id-label pairs from pickle.")

    except FileNotFoundError:
        logger.error(f"Pickle label file not found: {pickle_label_path}")
        return
    except Exception as e:
        logger.error(f"Error loading or processing pickle file {pickle_label_path}: {e}")
        return

    logger.info(f"Loading embedding paths from: {embeddings_manifest_path}")
    try:
        embeddings_df = pd.read_csv(embeddings_manifest_path)
        if 'talk_id' not in embeddings_df.columns:
            logger.error(f"Embeddings manifest {embeddings_manifest_path} missing 'talk_id' column.")
            return
        # Ensure talk_id is string for merging and strip prefix if present
        embeddings_df['talk_id'] = embeddings_df['talk_id'].astype(str)
        # Strip "urfunny_" prefix if it exists to match pickle keys
        embeddings_df['talk_id'] = embeddings_df['talk_id'].str.replace('^urfunny_', '', regex=True)
        
        # Drop the existing 'label' column from embeddings_df if it exists, as we're using labels from pickle
        if 'label' in embeddings_df.columns:
            embeddings_df = embeddings_df.drop(columns=['label'])
        logger.info(f"Loaded {len(embeddings_df)} entries from embeddings manifest.")
    except FileNotFoundError:
        logger.error(f"Embeddings manifest not found: {embeddings_manifest_path}")
        return
    except Exception as e:
        logger.error(f"Error reading embeddings manifest {embeddings_manifest_path}: {e}")
        return

    # Merge labels from pickle into the embeddings dataframe using 'talk_id'
    # This ensures we only keep entries for which we have embeddings
    # and get their labels from the authoritative pickle source.
    merged_df = pd.merge(embeddings_df, label_df, on='talk_id', how='inner')
    logger.info(f"After merging with labels from pickle, {len(merged_df)} entries remain.")
    
    if merged_df.empty:
        logger.error("No common talk_ids found between embeddings manifest and pickle labels. Cannot proceed.")
        return
        
    # Ensure label is integer
    merged_df['label'] = merged_df['label'].astype(int)

    logger.info(f"Generating train/val/test splits for {len(merged_df)} entries.")
    
    talk_ids = merged_df['talk_id']
    labels_for_split = merged_df['label']

    # Stratified split: First, split into train and a temporary set (val + test)
    # Ensure there are enough samples for stratification, especially for minority classes
    if len(labels_for_split.unique()) < 2 or labels_for_split.value_counts().min() < 2 : # Need at least 2 samples of each class for stratification
        logger.warning("Not enough samples for stratified split or only one class present. Using non-stratified split.")
        train_ids, temp_ids = train_test_split(
            talk_ids, test_size=(val_ratio + test_ratio), random_state=random_seed
        )
        # For non-stratified, we don't need temp_labels for the second split if we just split IDs
        if len(temp_ids) > 0 and (val_ratio + test_ratio) > 0:
             relative_val_ratio = val_ratio / (val_ratio + test_ratio) if (val_ratio + test_ratio) > 0 else 0
             if len(temp_ids) > 1 and relative_val_ratio < 1.0 : # Need at least 2 samples to split temp
                val_ids, test_ids = train_test_split(
                    temp_ids, test_size=(1.0 - relative_val_ratio), random_state=random_seed
                )
             elif len(temp_ids) == 1 and relative_val_ratio == 1.0: # all to val
                val_ids = temp_ids
                test_ids = pd.Series(dtype='object')
             elif len(temp_ids) == 1 and relative_val_ratio == 0.0: # all to test
                test_ids = temp_ids
                val_ids = pd.Series(dtype='object')
             else: # only one sample in temp, assign to val or test based on ratio, or default to val
                val_ids = temp_ids if relative_val_ratio > 0 else pd.Series(dtype='object')
                test_ids = temp_ids if relative_val_ratio == 0 else pd.Series(dtype='object')

        else:
            val_ids = pd.Series(dtype='object')
            test_ids = pd.Series(dtype='object')

    else: # Stratified split
        train_ids, temp_ids, _, temp_labels = train_test_split(
            talk_ids, labels_for_split, test_size=(val_ratio + test_ratio), random_state=random_seed, stratify=labels_for_split
        )
        if len(temp_ids) > 0 and (val_ratio + test_ratio) > 0:
            relative_val_ratio = val_ratio / (val_ratio + test_ratio)
            # Ensure temp_labels has enough samples of each class for stratification
            if len(temp_labels.unique()) < 2 or temp_labels.value_counts().min() < 2 or len(temp_ids) < 2:
                 logger.warning("Not enough samples in temp set for stratified val/test split. Using non-stratified for val/test.")
                 if len(temp_ids) > 1 and relative_val_ratio < 1.0 :
                     val_ids, test_ids = train_test_split(temp_ids, test_size=(1.0 - relative_val_ratio), random_state=random_seed)
                 elif len(temp_ids) == 1 and relative_val_ratio == 1.0:
                     val_ids = temp_ids; test_ids = pd.Series(dtype='object')
                 elif len(temp_ids) == 1 and relative_val_ratio == 0.0:
                     test_ids = temp_ids; val_ids = pd.Series(dtype='object')
                 else:
                     val_ids = temp_ids if relative_val_ratio > 0 else pd.Series(dtype='object')
                     test_ids = temp_ids if relative_val_ratio == 0 else pd.Series(dtype='object')
            else:
                val_ids, test_ids, _, _ = train_test_split(
                    temp_ids, temp_labels, test_size=(1.0 - relative_val_ratio), random_state=random_seed, stratify=temp_labels
                )
        else:
            val_ids = pd.Series(dtype='object')
            test_ids = pd.Series(dtype='object')


    split_map = {tid: 'train' for tid in train_ids}
    split_map.update({tid: 'val' for tid in val_ids})
    split_map.update({tid: 'test' for tid in test_ids})

    merged_df['split'] = merged_df['talk_id'].map(split_map)
    
    # Handle any talk_ids that might not have been assigned a split (should be rare if logic is correct)
    unassigned_splits = merged_df['split'].isnull().sum()
    if unassigned_splits > 0:
        logger.warning(f"{unassigned_splits} entries were not assigned a split. This should not happen. Assigning them to 'train'.")
        merged_df['split'].fillna('train', inplace=True)


    logger.info(f"Split counts:\n{merged_df['split'].value_counts(dropna=False)}")
    
    final_columns = ['talk_id', 'text_embedding_path', 'audio_embedding_path', 'video_embedding_path', 'label', 'split']
    # Ensure all original embedding manifest columns (except the old label) are preserved if they existed
    original_embedding_cols = [col for col in embeddings_df.columns if col not in ['talk_id', 'text_embedding_path', 'audio_embedding_path', 'video_embedding_path']]
    for col in original_embedding_cols:
        if col not in final_columns:
            final_columns.append(col)
            
    final_df = merged_df[final_columns]

    os.makedirs(os.path.dirname(output_manifest_path), exist_ok=True)
    logger.info(f"Saving final comprehensive manifest to: {output_manifest_path}")
    try:
        final_df.to_csv(output_manifest_path, index=False)
        logger.info("Final manifest generated successfully.")
    except Exception as e:
        logger.error(f"Error writing final manifest: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a comprehensive multimodal manifest with labels from pickle and new splits.")
    parser.add_argument("--pickle_labels", required=True, help="Path to the pickle file (e.g., humor_label_sdk.pkl).")
    parser.add_argument("--embeddings_manifest", required=True, help="Path to the CSV manifest with talk_id and embedding paths.")
    parser.add_argument("--output_manifest", required=True, help="Path to save the final comprehensive manifest CSV.")
    parser.add_argument("--train_ratio", type=float, default=0.70, help="Proportion for training set.")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Proportion for validation set.")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Proportion for test set.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for splits.")
    
    args = parser.parse_args()

    create_full_dataset_manifest(args.pickle_labels, args.embeddings_manifest, args.output_manifest,
                                 args.train_ratio, args.val_ratio, args.test_ratio, args.random_seed)
