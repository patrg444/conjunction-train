import pandas as pd
import os
import sys

def augment_manifest(manifest_path, output_path):
    """Adds transcript column to a manifest CSV with dummy text."""
    print(f"Augmenting manifest: {manifest_path}")
    try:
        df = pd.read_csv(manifest_path)
        print(f"Read {len(df)} rows.")

        # Generate dummy transcripts based on the sample ID
        # In a real scenario, this would come from an actual transcript file
        df['transcript'] = df['id'].apply(lambda id_val: f"This is a dummy transcript for sample {id_val}. {'It contains humor.' if df.loc[df['id'] == id_val, 'label'].values[0] == 1 else 'It does not contain humor.'}")

        # Save augmented manifest
        df.to_csv(output_path, index=False)
        print(f"Saved augmented manifest to: {output_path}")
        return True

    except FileNotFoundError:
        print(f"Error: Manifest file not found at {manifest_path}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"An unexpected error occurred during manifest augmentation: {e}", file=sys.stderr)
        return False


if __name__ == "__main__":
    # Define paths relative to the project root
    base_dir = os.getcwd()
    train_manifest_in = os.path.join(base_dir, "datasets/manifests/humor/train_humor.csv")
    val_manifest_in = os.path.join(base_dir, "datasets/manifests/humor/val_humor.csv")
    
    # Create output directories
    os.makedirs(os.path.join(base_dir, "datasets/manifests/humor"), exist_ok=True)
    
    train_manifest_out = os.path.join(base_dir, "datasets/manifests/humor/train_humor_with_text.csv")
    val_manifest_out = os.path.join(base_dir, "datasets/manifests/humor/val_humor_with_text.csv")

    # Augment manifests
    train_success = augment_manifest(train_manifest_in, train_manifest_out)
    val_success = augment_manifest(val_manifest_in, val_manifest_out)
    
    if train_success and val_success:
        print("Script finished successfully.")
        sys.exit(0)
    else:
        print("Script finished with errors.", file=sys.stderr)
        sys.exit(1)
