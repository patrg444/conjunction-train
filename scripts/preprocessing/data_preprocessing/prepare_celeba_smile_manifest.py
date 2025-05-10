import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_celeba_manifest(celeba_root, output_dir, val_split=0.1, random_state=42):
    """
    Prepares train and validation manifests for CelebA smiling classification.

    Args:
        celeba_root (str): Path to the root directory of the CelebA dataset
                           (containing 'img_align_celeba/' and 'list_attr_celeba.txt').
        output_dir (str): Directory to save the manifest CSV files.
        val_split (float): Proportion of the dataset to use for validation.
        random_state (int): Random state for reproducibility of the split.
    """
    img_dir = os.path.join(celeba_root, 'img_align_celeba')
    attr_file = os.path.join(celeba_root, 'list_attr_celeba.txt')

    # --- Input Validation ---
    if not os.path.isdir(img_dir):
        logging.error(f"Image directory not found: {img_dir}")
        return
    if not os.path.isfile(attr_file):
        logging.error(f"Attribute file not found: {attr_file}")
        return

    logging.info("Reading CelebA attribute data...")
    # Read attributes, skipping the first header line (count) and using space separation
    try:
        # Use regex-compatible separator to fix the '\s+' escape sequence warning
        attrs = pd.read_csv(attr_file, sep=r'\s+', header=1, engine='python')
        attrs['image_id'] = attrs.index # Use the index as the image filename
        attrs.index.name = 'image_id_idx' # Rename index column
        # Convert -1/1 labels to 0/1 for binary classification
        attrs = attrs.replace(-1, 0)
        logging.info(f"Loaded attributes for {len(attrs)} images.")
        logging.info(f"Available attributes: {list(attrs.columns)}")
    except Exception as e:
        logging.error(f"Error reading attribute file {attr_file}: {e}")
        return

    if 'Smiling' not in attrs.columns:
        logging.error("'Smiling' attribute not found in list_attr_celeba.txt. Check file format.")
        return

    # --- Filter for Smiling Attribute ---
    smile_data = attrs[['image_id', 'Smiling']].copy()
    smile_data.rename(columns={'Smiling': 'label'}, inplace=True)
    smile_data['image_path'] = smile_data['image_id'].apply(lambda x: os.path.join(img_dir, x))

    # --- Split Data ---
    logging.info(f"Splitting data into train/validation sets (validation split: {val_split})...")
    train_df, val_df = train_test_split(
        smile_data,
        test_size=val_split,
        random_state=random_state,
        stratify=smile_data['label'] # Stratify to maintain label distribution
    )

    # --- Save Manifests ---
    os.makedirs(output_dir, exist_ok=True)
    train_manifest_path = os.path.join(output_dir, 'train_smile.csv')
    val_manifest_path = os.path.join(output_dir, 'val_smile.csv')

    # Select only necessary columns
    train_df[['image_path', 'label']].to_csv(train_manifest_path, index=False)
    val_df[['image_path', 'label']].to_csv(val_manifest_path, index=False)

    logging.info(f"Train manifest saved to: {train_manifest_path} ({len(train_df)} samples)")
    logging.info(f"Validation manifest saved to: {val_manifest_path} ({len(val_df)} samples)")
    logging.info("Manifest preparation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare CelebA Smiling Manifests")
    parser.add_argument('--celeba_root', type=str, required=True,
                        help="Root directory of the CelebA dataset (containing img_align_celeba/, list_attr_celeba.txt, etc.)")
    parser.add_argument('--output_dir', type=str, default='datasets/manifests/humor',
                        help="Directory to save the output train_smile.csv and val_smile.csv files")
    parser.add_argument('--val_split', type=float, default=0.1,
                        help="Fraction of data to use for the validation set")
    parser.add_argument('--random_state', type=int, default=42,
                        help="Random state for train/val split reproducibility")

    args = parser.parse_args()

    prepare_celeba_manifest(args.celeba_root, args.output_dir, args.val_split, args.random_state)
