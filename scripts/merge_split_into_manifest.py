import pandas as pd
import argparse
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def merge_split_info(multimodal_manifest_path, label_split_manifest_path, output_path=None):
    """
    Merges the 'split' column from a label/split manifest into a multimodal embedding manifest.

    Args:
        multimodal_manifest_path (str): Path to the multimodal manifest CSV
                                        (must contain 'talk_id').
        label_split_manifest_path (str): Path to the manifest containing 'talk_id' and 'split'.
        output_path (str, optional): Path to save the merged manifest.
                                     If None, overwrites multimodal_manifest_path. Defaults to None.
    """
    if output_path is None:
        output_path = multimodal_manifest_path # Overwrite by default

    logger.info(f"Reading multimodal manifest from: {multimodal_manifest_path}")
    try:
        multi_df = pd.read_csv(multimodal_manifest_path)
        if 'talk_id' not in multi_df.columns:
            logger.error(f"Multimodal manifest missing 'talk_id' column.")
            return
    except FileNotFoundError:
        logger.error(f"Multimodal manifest not found at {multimodal_manifest_path}")
        return
    except Exception as e:
        logger.error(f"Error reading multimodal manifest: {e}")
        return

    logger.info(f"Reading label/split manifest from: {label_split_manifest_path}")
    try:
        label_df = pd.read_csv(label_split_manifest_path)
        if 'talk_id' not in label_df.columns or 'split' not in label_df.columns:
            logger.error(f"Label/split manifest missing 'talk_id' or 'split' column.")
            return
        # Keep only necessary columns to avoid duplicates after merge
        label_df = label_df[['talk_id', 'split']].drop_duplicates(subset=['talk_id'])
    except FileNotFoundError:
        logger.error(f"Label/split manifest not found at {label_split_manifest_path}")
        return
    except Exception as e:
        logger.error(f"Error reading label/split manifest: {e}")
        return

    logger.info(f"Merging 'split' column based on 'talk_id'. Initial rows: {len(multi_df)}")

    # Perform the merge
    # Use left merge to keep all rows from the multimodal manifest
    merged_df = pd.merge(multi_df, label_df, on='talk_id', how='left')

    # Check for rows that didn't get a split assigned (shouldn't happen if textonly is a subset)
    missing_split_count = merged_df['split'].isnull().sum()
    if missing_split_count > 0:
        logger.warning(f"{missing_split_count} entries in the multimodal manifest did not have a corresponding 'split' in the label manifest.")
        # Optionally drop rows without a split, or fill with a default like 'unknown'
        # For now, we keep them but they might cause issues downstream if not handled
        # merged_df = merged_df.dropna(subset=['split'])
        # logger.info(f"Dropped rows with missing splits. Final rows: {len(merged_df)}")

    logger.info(f"Merge complete. Final rows: {len(merged_df)}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info(f"Saving merged manifest to: {output_path}")
    try:
        merged_df.to_csv(output_path, index=False)
        logger.info("Merged manifest saved successfully.")
    except Exception as e:
        logger.error(f"Error writing merged manifest: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge split information into a multimodal manifest.")
    parser.add_argument("--multi_manifest", required=True,
                        help="Path to the multimodal embedding manifest CSV.")
    parser.add_argument("--label_split_manifest", required=True,
                        help="Path to the manifest CSV containing 'talk_id' and 'split'.")
    parser.add_argument("--output", default=None,
                        help="Path to save the merged manifest. If not provided, overwrites the multi_manifest.")

    args = parser.parse_args()
    merge_split_info(args.multi_manifest, args.label_split_manifest, args.output)
