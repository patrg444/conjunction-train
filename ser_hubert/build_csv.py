import os
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import glob

# --- Configuration ---
# Documented paths based on file system checks
DATASET_PATHS = {
    "crema_d": "data/CREMA-D", # Updated path based on list_files result
    "ravdess": "data/RAVDESS" # Path confirmed by list_files result
}

# Emotion mappings (adjust if dataset labels differ)
EMOTION_MAP_CREMA = {
    "ANG": "angry", "DIS": "disgust", "FEA": "fear",
    "HAP": "happy", "NEU": "neutral", "SAD": "sad"
}
EMOTION_MAP_RAVDESS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fear", "07": "disgust", "08": "surprise"
    # Note: 'calm' and 'surprise' might need mapping or exclusion
    # depending on your final LABEL_MAP in data_module.py
}

# RAVDESS speaker IDs for test set (as suggested in Appendix B)
RAVDESS_TEST_SPEAKERS = ["19", "20", "21", "22", "23", "24"]

def process_crema_d(crema_path):
    """Processes CREMA-D dataset."""
    print(f"Processing CREMA-D from: {crema_path}")
    # NOTE: Expecting .wav files after conversion from .flv or if correct path provided.
    filepaths = glob.glob(os.path.join(crema_path, "*.wav"))
    if not filepaths:
        print(f"Warning: No WAV files found in {crema_path}. Check path.")
        return []

    data = []
    for fp in filepaths:
        basename = os.path.basename(fp)
        parts = basename.split('_')
        if len(parts) < 4:
            print(f"Warning: Skipping malformed filename {basename}")
            continue
        speaker_id = parts[0]
        emotion_code = parts[2]
        emotion = EMOTION_MAP_CREMA.get(emotion_code)
        if emotion: # Only include emotions we have mapped
             data.append({
                 "path": fp, # Store full path initially
                 "speaker": f"crema_{speaker_id}", # Prefix to avoid ID collision
                 "emotion": emotion,
                 "dataset": "crema_d"
             })
        else:
            print(f"Warning: Skipping CREMA-D file with unmapped emotion code {emotion_code}: {basename}")
    print(f"Found {len(data)} valid CREMA-D files.")
    return data

def process_ravdess(ravdess_path):
    """Processes RAVDESS dataset."""
    print(f"Processing RAVDESS from: {ravdess_path}")
    # Path updated to directly look in Actor_* subdirs of the provided ravdess_path
    filepaths = glob.glob(os.path.join(ravdess_path, "Actor_*", "*.wav"))
    if not filepaths:
        print(f"Warning: No WAV files found in Actor_* subdirectories of {ravdess_path}. Check path.")
        return []

    data = []
    for fp in filepaths:
        basename = os.path.basename(fp)
        parts = basename.split('-')
        if len(parts) < 7:
             print(f"Warning: Skipping malformed filename {basename}")
             continue
        emotion_code = parts[2]
        speaker_id = parts[6].split('.')[0] # Get speaker ID before extension
        emotion = EMOTION_MAP_RAVDESS.get(emotion_code)

        # Exclude 'calm' and 'surprise' for this example, or map them if desired
        if emotion and emotion not in ["calm", "surprise"]:
            data.append({
                "path": fp, # Store full path initially
                "speaker": f"ravdess_{speaker_id}", # Prefix speaker ID
                "emotion": emotion,
                "dataset": "ravdess"
            })
        # else:
            # print(f"Skipping RAVDESS file with unmapped/excluded emotion {emotion}: {basename}")

    print(f"Found {len(data)} valid RAVDESS files (excluding calm/surprise).")
    return data

def main(args):
    all_data = []

    # Process each dataset if path is provided
    if os.path.exists(DATASET_PATHS["crema_d"]):
        all_data.extend(process_crema_d(DATASET_PATHS["crema_d"]))
    else:
        print(f"Path not found for CREMA-D: {DATASET_PATHS['crema_d']}")

    if os.path.exists(DATASET_PATHS["ravdess"]):
        all_data.extend(process_ravdess(DATASET_PATHS["ravdess"]))
    else:
        print(f"Path not found for RAVDESS: {DATASET_PATHS['ravdess']}")

    if not all_data:
        print("Error: No data found from any dataset path. Exiting.")
        return

    df = pd.DataFrame(all_data)
    print(f"\nTotal combined samples: {len(df)}")
    print("Emotion distribution:")
    print(df['emotion'].value_counts())

    # --- Splitting Logic ---
    # 1. Separate RAVDESS test speakers
    ravdess_test_speaker_ids = [f"ravdess_{spk}" for spk in RAVDESS_TEST_SPEAKERS]
    test_df = df[df['speaker'].isin(ravdess_test_speaker_ids)].copy()
    remaining_df = df[~df['speaker'].isin(ravdess_test_speaker_ids)].copy()

    print(f"\nUsing {len(test_df)} samples from RAVDESS speakers {RAVDESS_TEST_SPEAKERS} for TEST set.")
    print(f"{len(remaining_df)} samples remaining for train/validation split.")

    if len(remaining_df) == 0:
        print("Error: No remaining data for training and validation after separating test set. Check dataset paths and speaker IDs.")
        return

    # 2. Split remaining data into train and validation (stratified by speaker and emotion)
    # We need a combined key for stratification if possible, or stratify mainly by speaker
    # Stratifying by speaker ensures speaker independence between train/val
    train_df, val_df = train_test_split(
        remaining_df,
        test_size=args.val_size,
        random_state=args.random_seed,
        stratify=remaining_df['speaker'] # Stratify by speaker first
        # Add 'emotion' to stratify if speaker counts allow: stratify=remaining_df[['speaker', 'emotion']]
    )

    print(f"\nTrain set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")

    # --- Save CSVs ---
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Optional: Convert paths to relative if desired, assuming a base data dir
    # base_data_dir = "data" # Or args.base_data_dir
    # train_df['path'] = train_df['path'].apply(lambda x: os.path.relpath(x, base_data_dir))
    # val_df['path'] = val_df['path'].apply(lambda x: os.path.relpath(x, base_data_dir))
    # test_df['path'] = test_df['path'].apply(lambda x: os.path.relpath(x, base_data_dir))

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print(f"\nCSV files saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Train/Val/Test CSV splits for SER datasets.")
    parser.add_argument("--output_dir", type=str, default="splits", help="Directory to save the output CSV files.")
    parser.add_argument("--val_size", type=float, default=0.15, help="Proportion of non-test data to use for validation.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for train/val split.")
    # Add arguments for dataset paths if you want them configurable
    # parser.add_argument("--crema_path", type=str, default=DATASET_PATHS["crema_d"])
    # parser.add_argument("--ravdess_path", type=str, default=DATASET_PATHS["ravdess"])

    args = parser.parse_args()

    # Update DATASET_PATHS if args are provided (example)
    # if hasattr(args, 'crema_path'): DATASET_PATHS["crema_d"] = args.crema_path
    # if hasattr(args, 'ravdess_path'): DATASET_PATHS["ravdess"] = args.ravdess_path

    main(args)
