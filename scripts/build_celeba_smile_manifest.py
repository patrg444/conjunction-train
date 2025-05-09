import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split

def create_celeba_smile_manifests(attr_file_path, img_dir_root, ec2_dataset_root, output_dir, train_split_ratio=0.8, random_seed=42):
    """
    Creates train and validation manifests for CelebA smile detection.

    Args:
        attr_file_path (str): Path to the list_attr_celeba.txt file.
        img_dir_root (str): The root directory on the EC2 instance where CelebA images are stored
                             (e.g., '/home/ubuntu/datasets/celeba/imgs/img_align_celeba').
                             The manifest will contain paths relative to this root's parent
                             (e.g., 'celeba/imgs/img_align_celeba/000001.jpg').
        output_dir (str): Directory on the EC2 instance to save the manifest CSVs.
        train_split_ratio (float): Proportion of data to use for training.
        random_seed (int): Random seed for reproducibility.
    """
    print(f"Reading attributes from: {attr_file_path}")
    try:
        # Read the attribute file, skipping the first line (header count)
        # The actual header is on the second line
        df_attr = pd.read_csv(attr_file_path, sep='\s+', header=1)
        df_attr.index.name = 'image_id' # The first column is the image ID
        df_attr.reset_index(inplace=True)
        print(f"Read {len(df_attr)} records.")

        # Check if 'Smiling' column exists
        if 'Smiling' not in df_attr.columns:
            print(f"Error: 'Smiling' attribute column not found in {attr_file_path}", file=sys.stderr)
            sys.exit(1)

        print("Processing attributes...")
        # Create the manifest DataFrame
        manifest_data = []
        # Construct the relative path prefix based on img_dir_root relative to ec2_dataset_root
        # Example: if img_dir_root is /home/ubuntu/datasets/celeba/imgs/img_align_celeba
        # and ec2_dataset_root is /home/ubuntu/datasets
        # We want paths like celeba/imgs/img_align_celeba/000001.jpg
        relative_prefix = os.path.relpath(img_dir_root, ec2_dataset_root)
        print(f"Calculated relative prefix for manifest: {relative_prefix}")

        for index, row in df_attr.iterrows():
            image_id = row['image_id']
            smile_attr = row['Smiling']
            smile_label = 1 if smile_attr == 1 else 0 # Map -1 (no smile) to 0, 1 (smile) to 1
            rel_video_path = os.path.join(relative_prefix, image_id) # Construct relative path
            manifest_data.append({'rel_video': rel_video_path, 'smile_label': smile_label})

        df_manifest = pd.DataFrame(manifest_data)
        print(f"Created manifest DataFrame with {len(df_manifest)} entries.")

        # Split data
        print(f"Splitting data with ratio {train_split_ratio} and seed {random_seed}...")
        df_train, df_val = train_test_split(
            df_manifest,
            train_size=train_split_ratio,
            random_state=random_seed,
            stratify=df_manifest['smile_label'] # Stratify to maintain label distribution
        )
        print(f"Train size: {len(df_train)}, Validation size: {len(df_val)}")

        # Save manifests
        train_output_path = os.path.join(output_dir, "celeba_smile_train.csv")
        val_output_path = os.path.join(output_dir, "celeba_smile_val.csv")

        os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

        df_train.to_csv(train_output_path, index=False)
        print(f"Saved training manifest to: {train_output_path}")
        df_val.to_csv(val_output_path, index=False)
        print(f"Saved validation manifest to: {val_output_path}")

    except FileNotFoundError:
        print(f"Error: Attribute file not found at {attr_file_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # --- Configuration for EC2 instance ---
    ec2_base_dir = "/home/ubuntu/conjunction-train"
    ec2_dataset_root = "/home/ubuntu/datasets" # Parent of celeba dir

    celeba_attr_file = os.path.join(ec2_dataset_root, "celeba/list_attr_celeba.txt")
    celeba_img_dir = os.path.join(ec2_dataset_root, "celeba/imgs/img_align_celeba")
    output_manifest_dir = os.path.join(ec2_base_dir, "datasets") # Save manifests alongside humor ones

    print("Starting CelebA smile manifest creation...")
    create_celeba_smile_manifests(
        attr_file_path=celeba_attr_file,
        img_dir_root=celeba_img_dir,
        ec2_dataset_root=ec2_dataset_root, # Pass dataset root
        output_dir=output_manifest_dir
    )
    print("Script finished.")
