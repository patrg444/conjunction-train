import os
import pickle
import pandas as pd
from tqdm import tqdm

def build_manifests_from_pickles():
    """
    Builds train and validation manifests for UR-FUNNY dataset
    from provided pickle files and video file names, including transcripts.
    Outputs relative audio paths.
    """
    print(f"Current working directory: {os.getcwd()}") # Print current working directory

    # Absolute paths for input data (pickle files and videos)
    pickle_dir = "/home/ubuntu/conjunction-train/datasets/humor_datasets/ur_funny"
    video_dir = "/home/ubuntu/conjunction-train/datasets/humor/urfunny/raw/urfunny2_videos"

    # Absolute paths for output directories (manifests and audio)
    manifest_dir = "/home/ubuntu/conjunction-train/datasets/manifests/humor"
    audio_output_dir = "/home/ubuntu/conjunction-train/datasets/humor_datasets/ur_funny/audio" # Absolute path for audio output

    data_folds_path = os.path.join(pickle_dir, "data_folds.pkl")
    humor_label_path = os.path.join(pickle_dir, "humor_label_sdk.pkl")
    language_sdk_path = os.path.join(pickle_dir, "language_sdk.pkl") # Path to language_sdk.pkl

    os.makedirs(manifest_dir, exist_ok=True)
    os.makedirs(audio_output_dir, exist_ok=True)

    # Check if pickle files exist
    if not os.path.exists(data_folds_path):
        print(f"Error: data_folds.pkl not found at {data_folds_path}")
        print("Please ensure the UR-FUNNY pickle files are downloaded.")
        return {}
    if not os.path.exists(humor_label_path):
        print(f"Error: humor_label_sdk.pkl not found at {humor_label_path}")
        print("Please ensure the UR-FUNNY pickle files are downloaded.")
        return {}
    if not os.path.exists(language_sdk_path):
        print(f"Error: language_sdk.pkl not found at {language_sdk_path}")
        print("Please ensure the UR-FUNNY pickle files are downloaded.")
        return {}
    # Check if the real video directory exists
    if not os.path.exists(video_dir):
         print(f"Error: video directory not found at {video_dir}")
         print("Please ensure the UR-FUNNY videos are unzipped and the symbolic link is correct.")
         return {}


    print(f"Loading data folds from {data_folds_path}...")
    with open(data_folds_path, 'rb') as f:
        data_folds = pickle.load(f)

    print(f"Loading humor labels from {humor_label_path}...")
    with open(humor_label_path, 'rb') as f:
        humor_labels = pickle.load(f)

    print(f"Loading language data from {language_sdk_path}...")
    with open(language_sdk_path, 'rb') as f:
        language_data = pickle.load(f)

    train_ids = data_folds.get('train', [])
    val_ids = data_folds.get('dev', [])
    test_ids = data_folds.get('test', []) # Get test IDs

    train_data = []
    val_data = []
    test_data = [] # Initialize test data list

    print("Building train manifest...")
    for video_id in tqdm(train_ids):
        label = humor_labels.get(video_id, None)
        transcript = language_data.get(video_id, {}).get('text', '') # Get transcript, default to empty string
        # Corrected video_path construction
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        audio_path = os.path.join(audio_output_dir, f"{video_id}.wav") # Full path to check existence

        if label is not None and os.path.exists(video_path) and os.path.exists(audio_path):
            train_data.append({
                "talk_id": f"urfunny_{video_id}",
                "title": "UR-FUNNY",
                "text": transcript, # Include transcript
                "label": int(label),
                "rel_audio": f"{video_id}.wav", # Use 'rel_audio' with relative path
                "video_path": video_path,
                "split": "train" # Add split column
            })
        elif not os.path.exists(video_path):
            print(f"Warning: Video file not found for ID {video_id} at {video_path}. Skipping.")
        elif not os.path.exists(audio_path):
            print(f"Warning: Audio file not found for ID {video_id} at {audio_path}. Skipping.")


    print("Building validation manifest...")
    for video_id in tqdm(val_ids):
        label = humor_labels.get(video_id, None)
        transcript = language_data.get(video_id, {}).get('text', '') # Get transcript, default to empty string
        # Corrected video_path construction
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        audio_path = os.path.join(audio_output_dir, f"{video_id}.wav") # Full path to check existence

        if label is not None and os.path.exists(video_path) and os.path.exists(audio_path):
            val_data.append({
                "talk_id": f"urfunny_{video_id}",
                "title": "UR-FUNNY",
                "text": transcript, # Include transcript
                "label": int(label),
                "rel_audio": f"{video_id}.wav", # Use 'rel_audio' with relative path
                "video_path": video_path,
                "split": "val" # Add split column
            })
        elif not os.path.exists(video_path):
            print(f"Warning: Video file not found for ID {video_id} at {video_path}. Skipping.")
        elif not os.path.exists(audio_path):
            print(f"Warning: Audio file not found for ID {video_id} at {audio_path}. Skipping.")

    print("Building test manifest...") # Add test manifest building
    for video_id in tqdm(test_ids):
        label = humor_labels.get(video_id, None)
        transcript = language_data.get(video_id, {}).get('text', '')
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        audio_path = os.path.join(audio_output_dir, f"{video_id}.wav") # Full path to check existence

        if label is not None and os.path.exists(video_path) and os.path.exists(audio_path):
            test_data.append({
                "talk_id": f"urfunny_{video_id}",
                "title": "UR-FUNNY",
                "text": transcript,
                "label": int(label),
                "rel_audio": f"{video_id}.wav", # Use 'rel_audio' with relative path
                "video_path": video_path,
                "split": "test" # Add split column
            })
        elif not os.path.exists(video_path):
            print(f"Warning: Video file not found for ID {video_id} at {video_path}. Skipping.")
        elif not os.path.exists(audio_path):
            print(f"Warning: Audio file not found for ID {video_id} at {audio_path}. Skipping.")

    output_files = {}

    if train_data:
        train_output = os.path.join(manifest_dir, "ur_funny_train_humor.csv")
        train_df = pd.DataFrame(train_data)
        train_df.to_csv(train_output, index=False)
        output_files["train"] = train_output
        print(f"Created train manifest: {train_output} with {len(train_data)} samples.")

    if val_data:
        val_output = os.path.join(manifest_dir, "ur_funny_val_humor.csv")
        val_df = pd.DataFrame(val_data)
        val_df.to_csv(val_output, index=False)
        output_files["val"] = val_output
        print(f"Created validation manifest: {val_output} with {len(val_data)} samples.")

    if test_data: # Add test manifest writing
        test_output = os.path.join(manifest_dir, "ur_funny_test_humor.csv")
        test_df = pd.DataFrame(test_data)
        test_df.to_csv(test_output, index=False)
        output_files["test"] = test_output
        print(f"Created test manifest: {test_output} with {len(test_data)} samples.")

    return output_files

if __name__ == "__main__":
    build_manifests_from_pickles()
