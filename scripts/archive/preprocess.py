import os
import re
import numpy as np
import pandas as pd
from deepface import DeepFace
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import subprocess
import arff
import cv2
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("preprocess.log"),
        logging.StreamHandler()
    ]
)


def extract_video_embeddings(video_path, model_name="VGG-Face", num_frames=5):
    """Extracts video embeddings by averaging embeddings from multiple frames.

    Args:
        video_path: Path to the video file.
        model_name: Name of the DeepFace model to use.
        num_frames: Number of frames to extract.

    Returns:
        A NumPy array representing the averaged video embedding.
    """
    positions = np.linspace(0.1, 0.9, num_frames)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Failed to open video: {video_path}")
        return np.zeros(2622)  # Return zeros if video can't be opened

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    embeddings = []

    for pos in positions:
        frame_num = int(frame_count * pos)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        success, frame = cap.read()
        if success:
            try:
                embedding_objs = DeepFace.represent(img_path=frame, model_name=model_name, enforce_detection=False)
                primary_embedding = select_primary_face(embedding_objs)
                embeddings.append(primary_embedding)
            except Exception as e:
                logging.error(f"DeepFace failed at frame {frame_num} of video {video_path}: {e}")
                embeddings.append(np.zeros(2622))  # or appropriate size for model
        else:
            logging.warning(f"Failed to extract frame {frame_num} from {video_path}")
            embeddings.append(np.zeros(2622))

    cap.release()
    return np.mean(embeddings, axis=0)  # Average embedding across frames

def select_primary_face(embedding_objs):
    """Selects the primary face embedding from a list of DeepFace embeddings.

    Args:
        embedding_objs: A list of embedding objects returned by DeepFace.represent.

    Returns:
        The embedding of the primary face (largest bounding box area).
    """
    # Select the face with the largest bounding box area
    if len(embedding_objs) == 1:
        return embedding_objs[0]["embedding"]

    areas = []
    for obj in embedding_objs:
        x, y, w, h = obj["facial_area"].values()
        areas.append((w * h, obj["embedding"]))
    primary_embedding = max(areas, key=lambda x: x[0])[1]
    return primary_embedding

def extract_video_features(video_paths, model_name="VGG-Face"):
    """Extracts video features using DeepFace.

    Args:
        video_paths: List of paths to video files.
        model_name: Name of the DeepFace model to use.

    Returns:
        A NumPy array of video features.
    """
    features = []
    for video_path in tqdm(video_paths, desc=f"Extracting video features ({model_name})"):
        logging.info(f"Extracting features for video: {video_path}")
        try:
            embedding = extract_video_embeddings(video_path, model_name=model_name)
            features.append(embedding)
        except Exception as e:
            logging.error(f"Error processing {video_path}: {e}")
            features.append(np.zeros(2622))  # Append zero vector on error

    return np.array(features)

def extract_audio_features(audio_paths, config_file="opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/config/egemaps/v02/eGeMAPSv02.conf", opensmile_path="/Users/patrickgloria/conjunction-train/opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/bin/SMILExtract", output_dir="temp_audio_features"):
    """Extracts audio features using openSMILE.

    Args:
        audio_paths: List of paths to audio files.
        config_file: Path to the openSMILE configuration file.
        opensmile_path: Path to the openSMILE executable.
        output_dir: Directory to store temporary ARFF files.

    Returns:
        NumPy array of audio features.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for audio_path in tqdm(audio_paths, desc="Extracting audio features (openSMILE)"):
        audio_basename = os.path.basename(audio_path)
        output_file = os.path.join(output_dir, f"{os.path.splitext(audio_basename)[0]}_egemaps.arff")
        
        # Run openSMILE using the same command format that works in multimodal_preprocess.py
        command = [
            opensmile_path,
            "-C", config_file,
            "-I", audio_path,
            "-O", output_file,
            "-instname", audio_basename,
            "-loglevel", "1"
        ]
        
        logging.info(f"Running openSMILE command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            logging.error(f"Error running openSMILE: {result.stderr}")
            # Create a temp directory to store audio files for this path
            temp_arff_dir = os.path.join(output_dir, f"{os.path.splitext(audio_basename)[0]}_temp")
            os.makedirs(temp_arff_dir, exist_ok=True)

    # Parse ARFF files using our custom parser that handles openSMILE format
    features, timestamps = load_arff_features(output_dir)
    
    if features.size > 0:
        logging.info(f"Extracted {features.shape[0]} audio feature vectors with {features.shape[1]} dimensions")
    else:
        logging.warning("Failed to extract audio features")
        
    # For backward compatibility, just return the features
    return features

# Function moved to utils.py to provide reusable ARFF parser for both preprocess.py and multimodal_preprocess.py
# Using import instead
from utils import load_arff_features

def synchronize_and_save_data(audio_features, video_features, audio_paths, video_paths, labels, output_prefix):
    """Synchronizes audio and video features, normalizes them, and saves the results.

    Args:
        audio_features: NumPy array of audio features.
        video_features: NumPy array of video features.
        audio_paths: List of paths to the original audio files.
        video_paths: List of paths to the original video files.
        labels: List of emotion labels.
        output_prefix: Prefix for output files (e.g., "train", "val", "test").
    """

    # --- Label Conversion (CREMA-D to RAVDESS format) ---
    label_mapping = {
        "ANG": 1,  # Angry
        "DIS": 2,  # Disgust
        "FEA": 3,  # Fear
        "HAP": 4,  # Happy
        "SAD": 5,  # Sad
        "NEU": 6,  # Neutral
    }
    # RAVDESS also has 'calm' (2) and 'surprised' (8), but CREMA-D doesn't.

    numeric_labels = []
    for label in labels:
        if isinstance(label, int):  # RAVDESS label (already numeric)
            numeric_labels.append(label)
        elif isinstance(label, str):  # CREMA-D label (string)
            numeric_labels.append(label_mapping.get(label, 0))  # 0 for unknown
        else:
            numeric_labels.append(0) # Handle unexpected label types

    # --- Synchronization (File-Level) ---
    # Duplicate audio features for each video frame within a file.
    synchronized_audio_features = []
    synchronized_video_features = []
    synchronized_labels = []

    audio_path_dict = {}
    for i, path in enumerate(audio_paths):
        audio_path_dict[os.path.splitext(os.path.basename(path))[0]] = i

    video_path_dict = {}
    for i, path in enumerate(video_paths):
        video_path_dict[os.path.splitext(os.path.basename(path))[0]] = i

    # Iterate by video file, since that's our frame-level resolution
    for video_idx, video_path in enumerate(video_paths):
        video_base = os.path.splitext(os.path.basename(video_path))[0]
        
        # Find corresponding audio file index
        audio_idx = audio_path_dict.get(video_base)

        if audio_idx is not None:
            # Get number of video frames (approximated by length of video feature vector)
            num_video_frames = 1 # we have one embedding per video
            
            # Duplicate audio features for each video frame
            synchronized_audio_features.extend([audio_features[audio_idx]] * num_video_frames)
            synchronized_video_features.append(video_features[video_idx]) # one embedding per video
            synchronized_labels.extend([numeric_labels[video_idx]] * num_video_frames)

    synchronized_audio_features = np.array(synchronized_audio_features)
    synchronized_video_features = np.array(synchronized_video_features)
    synchronized_labels = np.array(synchronized_labels)

    # --- Standardization ---
    audio_mean = np.mean(synchronized_audio_features, axis=0)
    audio_std = np.std(synchronized_audio_features, axis=0)
    video_mean = np.mean(synchronized_video_features, axis=0)
    video_std = np.std(synchronized_video_features, axis=0)

    # Avoid division by zero
    audio_std[audio_std == 0] = 1
    video_std[video_std == 0] = 1

    normalized_audio_features = (synchronized_audio_features - audio_mean) / audio_std
    normalized_video_features = (synchronized_video_features - video_mean) / video_std

    # --- Save Data ---
    np.save(f"{output_prefix}_audio_features.npy", normalized_audio_features)
    np.save(f"{output_prefix}_video_features.npy", normalized_video_features)
    np.save(f"{output_prefix}_labels.npy", synchronized_labels)
    print(f"Saved preprocessed data to {output_prefix}_*.npy")

# --- RAVDESS ---
def load_ravdess_data(ravdess_dir):
    """Loads RAVDESS audio and video data and extracts labels.

    Args:
        ravdess_dir: Path to the RAVDESS directory.

    Returns:
        A tuple (audio_data, video_data, labels), where:
          audio_data: List of paths to audio files.
          video_data: List of paths to video files.
          labels: List of corresponding emotion labels (integers).
    """
    audio_data = []
    video_data = []
    audio_labels = []  # Separate labels for audio and video
    video_labels = []

    speech_dir = os.path.join(ravdess_dir, "Audio_Speech_Actors_01-24")
    video_dir = os.path.join(ravdess_dir, "Actor_24")  # Video files are ONLY in Actor_24

    if not os.path.exists(speech_dir):
        raise FileNotFoundError(f"RAVDESS speech directory not found: {speech_dir}")
    if not os.path.exists(video_dir):
        raise FileNotFoundError(f"RAVDESS video directory not found: {video_dir}")

    # Load audio data and labels
    for actor_dir in tqdm(sorted(os.listdir(speech_dir)), desc="Loading RAVDESS Audio"):
        if not actor_dir.startswith("Actor_"):
            continue
        actor_path = os.path.join(speech_dir, actor_dir)
        if not os.path.isdir(actor_path):
            continue

        for file in os.listdir(actor_path):
            if file.endswith(".wav"):
                file_path = os.path.join(actor_path, file)
                audio_data.append(file_path)
                match = re.match(r"(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)\.wav", file)
                if not match:
                    print(f"Warning: Could not parse filename {file}")
                    continue
                _, _, emotion, _, _, _, _ = match.groups()
                audio_labels.append(int(emotion))

    # Load video data
    for file in tqdm(sorted(os.listdir(video_dir)), desc="Loading RAVDESS Video"):
        if file.endswith(".mp4"):
            file_path = os.path.join(video_dir, file)
            video_data.append(file_path)
            match = re.match(r"(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)\.mp4", file)
            if not match:
                print(f"Warning: Could not parse filename {file}")
                continue
            _, _, emotion, _, _, _, _ = match.groups()
            video_labels.append(int(emotion))  # Add video label


    # Create dictionaries to map filenames to paths and labels
    audio_dict = {os.path.splitext(os.path.basename(path))[0]: (path, label) for path, label in zip(audio_data, audio_labels)}
    video_dict = {os.path.splitext(os.path.basename(path))[0]: (path, label) for path, label in zip(video_data, video_labels)}

    # Match audio and video files based on filenames
    matched_audio_data = []
    matched_video_data = []
    matched_labels = []

    for video_base, (video_path, video_label) in video_dict.items():
        audio_info = audio_dict.get(video_base)
        if audio_info:
            audio_path, audio_label = audio_info
            matched_audio_data.append(audio_path)
            matched_video_data.append(video_path)
            matched_labels.append(audio_label)  # Use audio label (should be the same)
        else:
            print(f"Warning: No matching audio file found for video: {video_path}")

    return matched_audio_data, matched_video_data, matched_labels

def load_crema_d_data(crema_d_dir):
    """Loads CREMA-D audio and video data and extracts labels.

    Args:
        crema_d_dir: Path to the CREMA-D directory.

    Returns:
        A tuple (audio_data, video_data, labels), where:
          audio_data: List of paths to audio files.
          video_data: List of paths to video files.
          labels: List of corresponding emotion labels (strings).
    """
    audio_data = []
    video_data = []
    labels = []

    audio_dir = os.path.join(crema_d_dir, "AudioWAV")
    video_dir = os.path.join(crema_d_dir, "VideoFlash")

    for file in tqdm(sorted(os.listdir(audio_dir)), desc="Loading CREMA-D Audio"):
        if not file.endswith(".wav"):
            continue

        file_path = os.path.join(audio_dir, file)
        match = re.match(r"(\d+)_([A-Z]+)_([A-Z]+)_([A-Z]+)", file) # e.g. 1001_DFA_ANG_XX.wav
        if not match:
            print(f"Warning: Could not parse filename {file}")
            continue
        
        _, _, emotion, _ = match.groups()
        audio_data.append(file_path)
        labels.append(emotion)

    # Assumes video files have the same names as audio files, but with .flv extension
    for file in tqdm(sorted(os.listdir(video_dir)), desc="Loading CREMA-D Video"):
        if not file.endswith(".flv"):
            continue
        
        file_path = os.path.join(video_dir, file)
        match = re.match(r"(\d+)_([A-Z]+)_([A-Z]+)_([A-Z]+)", file) # e.g. 1001_DFA_ANG_XX.flv
        if not match:
            print(f"Warning: Could not parse filename {file}")
            continue

        video_data.append(file_path)

    # Ensure that the number of audio and video files match
    if len(audio_data) != len(video_data):
        raise ValueError("Number of audio and video files in CREMA-D do not match.")
    
    # Ensure video labels are in the same order by creating a dict and looking up
    labels_dict = {os.path.basename(a).split('.')[0]: l for a, l in zip(audio_data, labels)}
    video_labels = [labels_dict[os.path.basename(v).split('.')[0]] for v in video_data]

    return audio_data, video_data, video_labels


if __name__ == '__main__':
    # --- Load Data ---
    ravdess_dir = "data/RAVDESS"
    ravdess_dir = "data/RAVDESS"
    crema_d_dir = "data/CREMA-D"

    # --- Check for required directories ---
    if not os.path.exists(ravdess_dir):
        raise FileNotFoundError(f"RAVDESS directory not found: {ravdess_dir}")
    if not os.path.exists(crema_d_dir):
        raise FileNotFoundError(f"CREMA-D directory not found: {crema_d_dir}")
    if not os.path.exists(os.path.join(crema_d_dir, "AudioWAV")):
        raise FileNotFoundError(f"CREMA-D AudioWAV directory not found: {os.path.join(crema_d_dir, 'AudioWAV')}")
    if not os.path.exists(os.path.join(crema_d_dir, "VideoFlash")):
        raise FileNotFoundError(f"CREMA-D VideoFlash directory not found: {os.path.join(crema_d_dir, 'VideoFlash')}")

    ravdess_audio, ravdess_video, ravdess_labels = load_ravdess_data(ravdess_dir)
    crema_d_audio, crema_d_video, crema_d_labels = load_crema_d_data(crema_d_dir)

    print(f"RAVDESS: Loaded {len(ravdess_audio)} audio files, {len(ravdess_video)} video files, {len(ravdess_labels)} labels.")
    print(f"CREMA-D: Loaded {len(crema_d_audio)} audio files, {len(crema_d_video)} video files, {len(crema_d_labels)} labels.")

    # --- Combine Data ---
    # For now, just concatenate. Later, handle potential label differences.
    all_audio = ravdess_audio + crema_d_audio
    all_video = ravdess_video + crema_d_video
    all_labels = ravdess_labels + crema_d_labels  # Will need to be careful about label types

    # --- Split Data ---
    # First, split into train+val and test sets
    audio_train_val, audio_test, video_train_val, video_test, labels_train_val, labels_test = train_test_split(
        all_audio, all_video, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )

    # Then, split train+val into train and val sets
    audio_train, audio_val, video_train, video_val, labels_train, labels_val = train_test_split(
        audio_train_val, video_train_val, labels_train_val, test_size=0.25, random_state=42, stratify=labels_train_val
    )  # 0.25 * 0.8 = 0.2

    print(f"Training set size: {len(audio_train)}")
    print(f"Validation set size: {len(audio_val)}")
    print(f"Test set size: {len(audio_test)}")

    # --- Extract Video Features ---
    video_features_train = extract_video_features(video_train)
    video_features_val = extract_video_features(video_val)
    video_features_test = extract_video_features(video_test)

    # --- Extract Audio Features ---
    opensmile_config = "config/eGeMAPSv02.conf"  # Relative path to config file
    audio_features_train = extract_audio_features(audio_train, opensmile_config, output_dir="temp_audio_features_train")
    audio_features_val = extract_audio_features(audio_val, opensmile_config,  output_dir="temp_audio_features_val")
    audio_features_test = extract_audio_features(audio_test, opensmile_config, output_dir="temp_audio_features_test")


    print(f"Audio features train shape: {audio_features_train.shape}")
    print(f"Audio features validation shape: {audio_features_val.shape}")
    print(f"Audio features test shape: {audio_features_test.shape}")

    # --- Synchronize and Save Data ---
    synchronize_and_save_data(audio_features_train, video_features_train, audio_train, video_train, labels_train, "train")
    synchronize_and_save_data(audio_features_val, video_features_val, audio_val, video_val, labels_val, "val")
    synchronize_and_save_data(audio_features_test, video_features_test, audio_test, video_test, labels_test, "test")

    # --- Temporary Testing Code ---
    print("--- Testing extract_video_embeddings ---")
    sample_video_path = "data/RAVDESS/Actor_24/03-01-01-01-01-01-24.mp4"
    sample_video_embedding = extract_video_embeddings(sample_video_path)
    print(f"Shape of sample video embedding: {sample_video_embedding.shape}")

    print("--- Testing extract_audio_features ---")
    sample_audio_path = "data/RAVDESS/Audio_Speech_Actors_01-24/Actor_24/03-01-01-01-01-01-24.wav"
    sample_audio_features = extract_audio_features([sample_audio_path], opensmile_config, output_dir="temp_test_audio")
    print(f"Shape of sample audio features: {sample_audio_features.shape}")
    # --- End Temporary Testing Code ---
