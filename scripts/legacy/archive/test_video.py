import os
import numpy as np
from deepface import DeepFace
from tqdm import tqdm
import cv2
import logging
from typing import List, Tuple  # Import for type hinting

from .utils import load_ravdess_video, load_crema_d_video  # Import ONLY video functions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("preprocess_video_test.log"),  # Log to file
        logging.StreamHandler()  # Log to console
    ]
)

def extract_video_embeddings(video_path: str, model_name: str = "VGG-Face", num_frames: int = 5) -> np.ndarray:
    """Extracts video embeddings from a single video.

    Args:
        video_path: Path to the video file.
        model_name: The name of the DeepFace model to use.
        num_frames: The number of frames to sample.

    Returns:
        A NumPy array representing the average video embedding, or a zero
        vector with dimensions matching the chosen model's embeddings if an error occurs.
    """
    positions = np.linspace(0.1, 0.9, num_frames)  # Sample frames evenly
    cap = cv2.VideoCapture(video_path)
    logged_errors = set()  # Keep track of logged errors to avoid duplicates

    if not cap.isOpened():
        error_msg = f"Failed to open video: {video_path}"
        if error_msg not in logged_errors:  # Log only once per error message
            logging.error(error_msg)
            logged_errors.add(error_msg)
        return np.zeros(get_embedding_dimension(model_name))  # Return zero vector with correct dimensions

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    embeddings: List[List[float]] = []  # Initialize the embeddings list

    for pos in positions:
        frame_num = int(frame_count * pos)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        success, frame = cap.read()

        if success:
            try:
                # Use DeepFace to represent the frame
                embedding_objs = DeepFace.represent(img_path=frame, model_name=model_name, enforce_detection=False)
                logging.info(f"Model: {model_name}, Embedding shape: {len(embedding_objs[0]['embedding'])}")
                primary_embedding = select_primary_face(embedding_objs)  # Select the largest face
                embeddings.append(primary_embedding)
            except Exception as e:
                error_msg = f"DeepFace failed at frame {frame_num} of video {video_path}: {type(e).__name__}: {e}"
                if error_msg not in logged_errors:
                    logging.error(error_msg)
                    logged_errors.add(error_msg)
                embeddings.append(np.zeros(get_embedding_dimension(model_name)))  # Append zero vector on DeepFace error, correct shape
        else:
            error_msg = f"Failed to extract frame {frame_num} from {video_path}"
            if error_msg not in logged_errors:
                logging.warning(error_msg)  # Warning for frame extraction failure
                logged_errors.add(error_msg)
            embeddings.append(np.zeros(get_embedding_dimension(model_name))) #Correct shape

    cap.release()  # Release the video capture object
    return np.mean(embeddings, axis=0)  # Return the average embedding


def select_primary_face(embedding_objs: List[dict]) -> List[float]:
    """Selects the primary (largest) face embedding from a list of embeddings.

    Args:
        embedding_objs: A list of embedding objects returned by DeepFace.represent.

    Returns:
        The embedding (a list of floats) of the primary face.
    """
    if len(embedding_objs) == 1:
        return embedding_objs[0]["embedding"]

    areas = [(obj["facial_area"]["w"] * obj["facial_area"]["h"], obj["embedding"]) for obj in embedding_objs]
    return max(areas, key=lambda x: x[0])[1]  # Return embedding of largest face


def extract_video_features(video_paths: List[str], model_name: str = "VGG-Face") -> np.ndarray:
    """Extracts video features for a list of video files.

    Args:
        video_paths: A list of video file paths.
        model_name: The DeepFace model name.

    Returns:
        A NumPy array where each row is the embedding of a video.
    """
    features = []
    logged_errors = set()  # Track errors to avoid duplicate logging
    total_videos = len(video_paths)

    for video_path in tqdm(video_paths, total=total_videos, desc=f"Extracting video features ({model_name})"):
        try:
            embedding = extract_video_embeddings(video_path, model_name=model_name)  # Using model_name param which matches DeepFace.represent API
            features.append(embedding)
        except Exception as e:
            error_msg = f"Error processing {video_path}: {type(e).__name__}: {e}"
            if error_msg not in logged_errors:
                logging.error(error_msg)
                logged_errors.add(error_msg)
            features.append(np.zeros(get_embedding_dimension(model_name))) #Correct shape

    return np.array(features)

def get_embedding_dimension(model_name: str) -> int:
    """Returns the embedding dimension for a given DeepFace model."""
    if model_name == "VGG-Face":
        return 4096  # Corrected from 2622 to 4096
    elif model_name == "Facenet":
        return 128
    elif model_name == "Facenet512":
        return 512
    elif model_name == "OpenFace":
        return 128
    elif model_name == "DeepFace": # default model
        return 4096
    else:
        raise ValueError(f"Unsupported model: {model_name}")

if __name__ == '__main__':
    # --- Load Data ---
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


    ravdess_video = load_ravdess_video(ravdess_dir)  # Load ONLY video
    # crema_d_video = load_crema_d_video(crema_d_dir)   # Load ONLY video (if needed)

    print(f"RAVDESS: Loaded {len(ravdess_video)} video files.")

    # --- Combine Data (Optional) ---
    # all_video = ravdess_video + crema_d_video  # Combine if you're using both datasets

    # --- Extract Video Features (Full pipeline - uncomment when ready) ---
    video_features = extract_video_features(ravdess_video, model_name="VGG-Face")
    print(f"Video features shape: {video_features.shape}")

    # --- Step 1: Minimal OpenCV Test (Keep this for focused debugging) ---
    #print("--- Testing OpenCV ---")
    #sample_video_path = "data/RAVDESS/Actor_24/02-01-08-02-02-02-24.avi"  # Converted AVI file
    #print(f"  Testing with video: {os.path.abspath(sample_video_path)}")  # Absolute path
    #cap = cv2.VideoCapture(sample_video_path)
    #if cap.isOpened():
    #    ret, frame = cap.read()
    #    if ret:
    #        print("  Successfully read a frame!")
    #    else:
    #        print("  Failed to read a frame.")
    #else:
    #    print("  Failed to open the video file.")
    #cap.release()

    # --- Step 2: Test extract_video_embeddings (Uncomment after OpenCV test passes) ---
    #print("\n--- Testing extract_video_embeddings ---")
    #sample_video_embedding = extract_video_embeddings(sample_video_path, model_name="VGG-Face") #Corrected function name.
    #print(f"  Shape of sample video embedding: {sample_video_embedding.shape}")
