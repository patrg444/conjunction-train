import pickle
import json
import os
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths based on the project structure
DATASET_BASE_DIR = "datasets/humor_datasets/ur_funny"
LANGUAGE_SDK_PATH = os.path.join(DATASET_BASE_DIR, "language_sdk.pkl")
HUMOR_LABEL_SDK_PATH = os.path.join(DATASET_BASE_DIR, "humor_label_sdk.pkl")
DATA_FOLDS_PATH = os.path.join(DATASET_BASE_DIR, "data_folds.pkl")
OUTPUT_JSON_PATH = os.path.join(DATASET_BASE_DIR, "ur_funny_final.json")

def load_pickle(filepath):
    """Loads data from a pickle file."""
    if not os.path.exists(filepath):
        logging.error(f"Pickle file not found: {filepath}")
        return None
    logging.info(f"Loading {filepath}...")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    logging.info(f"Loaded {filepath}")
    return data

def convert_pickles_to_json():
    """Converts UR-FUNNY pickle data into a single JSON file."""
    language_sdk = load_pickle(LANGUAGE_SDK_PATH)
    humor_label_sdk = load_pickle(HUMOR_LABEL_SDK_PATH)
    data_folds = load_pickle(DATA_FOLDS_PATH)

    if language_sdk is None or humor_label_sdk is None or data_folds is None:
        logging.error("Failed to load all required pickle files. Aborting conversion.")
        return False

    json_data = []
    # The keys in language_sdk and humor_label_sdk are the utterance IDs
    utterance_ids = list(language_sdk.keys())

    logging.info("Converting pickle data to JSON format...")
    for utt_id in tqdm(utterance_ids):
        lang_data = language_sdk.get(utt_id)
        humor_label = humor_label_sdk.get(utt_id)

        if lang_data is None or humor_label is None:
            logging.warning(f"Skipping utterance ID {utt_id} due to missing language or humor data.")
            continue

        # Extract video_id from utterance_id (assuming format like "Speaker_XXX_video_YYYYY_segment_ZZZ")
        # Or simply use the utterance_id as video_id if that's how it's linked to video files
        # Based on build_ur_funny_manifest.py, it seems the video files are named video_id.wav
        # Let's assume video_id is the part before the last underscore for now, or check language_sdk structure
        # Looking at the structure provided by the user, the keys are "unique id of each humor / not humor video utterance".
        # The build_ur_funny_manifest.py script uses video_id.wav for raw audio.
        # Let's assume the video_id is the part of the utterance_id before the last underscore.
        # Example: "Speaker_012_video_00073_segment_0" -> video_id "Speaker_012_video_00073"
        parts = utt_id.split('_')
        if len(parts) > 1:
             video_id = '_'.join(parts[:-1])
        else:
             video_id = utt_id # Fallback if format is unexpected


        # The 'sentence' field in the original JSON seems to correspond to the punchline sentence
        punchline_sentence = " ".join(lang_data.get('punchline_sentence', []))

        json_data.append({
            "video_id": video_id,
            "start_time": lang_data.get('punchline_intervals', [[0.0, 0.0]])[0][0], # Assuming start of first word
            "end_time": lang_data.get('punchline_intervals', [[0.0, 0.0]])[-1][-1], # Assuming end of last word
            "humor": int(humor_label), # Ensure label is integer
            "sentence": punchline_sentence # Use punchline sentence as transcript
        })

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)

    logging.info(f"Saving converted data to {OUTPUT_JSON_PATH}")
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(json_data, f, indent=2)

    logging.info(f"Successfully created {OUTPUT_JSON_PATH} with {len(json_data)} entries.")
    return True

if __name__ == "__main__":
    convert_pickles_to_json()
