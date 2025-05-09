import pandas as pd
import json
import os
import sys
import ast
from tqdm import tqdm

def extract_video_id(rel_path):
    """Extracts video ID from paths like '.../video_segments/1_4789_06.mp4'"""
    try:
        filename = os.path.basename(rel_path) # e.g., '1_4789_06.mp4'
        video_id_part = filename.split('_')[0] # e.g., '1'
        # Ensure it's not empty and potentially numeric if needed, though JSON keys are strings
        if video_id_part:
            return video_id_part
    except Exception as e:
        print(f"Warning: Could not extract video ID from path {rel_path}: {e}", file=sys.stderr)
    return None

def load_transcripts(json_path):
    """Loads the transcript JSON, handling potential string-encoded dicts."""
    print(f"Loading transcripts from {json_path}...")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        print(f"Loaded data for {len(data)} video IDs.")

        processed_data = {}
        for video_id, content in tqdm(data.items(), desc="Processing video transcripts"):
            if 'Video clips' in content:
                video_clips = content['Video clips']
                # Handle case where 'Video clips' might be stored as a string literal
                if isinstance(video_clips, str):
                    try:
                        video_clips = ast.literal_eval(video_clips)
                    except (ValueError, SyntaxError) as e:
                        print(f"Warning: Could not eval Video clips string for video {video_id}: {e}", file=sys.stderr)
                        video_clips = {} # Assign empty dict if parsing fails

                if isinstance(video_clips, dict):
                     # Sort by clip index (assuming keys are numeric strings '0', '1', ...)
                    sorted_clips = sorted(video_clips.items(), key=lambda item: int(item[0]))
                    all_utterances = " ".join(
                        clip_data.get('Utterance', '').strip()
                        for _, clip_data in sorted_clips if isinstance(clip_data, dict)
                    )
                    processed_data[video_id] = all_utterances.strip()
                else:
                     processed_data[video_id] = ""
                     print(f"Warning: Video clips for ID {video_id} was not a dict after potential eval.", file=sys.stderr)
            else:
                processed_data[video_id] = ""
                print(f"Warning: No 'Video clips' key found for video ID {video_id}", file=sys.stderr)

        print("Finished processing transcripts.")
        return processed_data
    except FileNotFoundError:
        print(f"Error: Transcript file not found at {json_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from {json_path}. {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during transcript loading: {e}", file=sys.stderr)
        sys.exit(1)


def augment_manifest(manifest_path, transcript_data, output_path):
    """Adds transcript column to a manifest CSV."""
    print(f"Augmenting manifest: {manifest_path}")
    try:
        df = pd.read_csv(manifest_path)
        print(f"Read {len(df)} rows.")

        # Extract video ID
        df['video_id'] = df['rel_video'].apply(extract_video_id)

        # Map transcripts
        # Use .get(id, "") to handle cases where video_id might be None or not in transcripts
        df['transcript'] = df['video_id'].apply(lambda vid: transcript_data.get(vid, ""))

        # Drop the temporary video_id column
        df = df.drop(columns=['video_id'])

        # Save augmented manifest
        df.to_csv(output_path, index=False)
        print(f"Saved augmented manifest to: {output_path}")

    except FileNotFoundError:
        print(f"Error: Manifest file not found at {manifest_path}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred during manifest augmentation: {e}", file=sys.stderr)


if __name__ == "__main__":
    # Define paths relative to the EC2 instance filesystem
    # Assuming script is run from /home/ubuntu/conjunction-train
    base_dir = "/home/ubuntu/conjunction-train"
    transcript_json_path = "/home/ubuntu/datasets/SMILE/raw/SMILE_DATASET/annotations/multimodal_textual_representation.json"
    train_manifest_in = os.path.join(base_dir, "datasets/humor_train.csv")
    val_manifest_in = os.path.join(base_dir, "datasets/humor_val.csv")
    train_manifest_out = os.path.join(base_dir, "datasets/humor_train_with_text.csv")
    val_manifest_out = os.path.join(base_dir, "datasets/humor_val_with_text.csv")

    # Create datasets directory if it doesn't exist (local equivalent on EC2)
    # This script doesn't run locally, but good practice if it did.
    # os.makedirs(os.path.dirname(train_manifest_out), exist_ok=True)

    # Load transcripts
    transcripts = load_transcripts(transcript_json_path)

    if transcripts:
        # Augment manifests
        augment_manifest(train_manifest_in, transcripts, train_manifest_out)
        augment_manifest(val_manifest_in, transcripts, val_manifest_out)
        print("Script finished.")
    else:
        print("Transcript loading failed, cannot augment manifests.", file=sys.stderr)
        sys.exit(1)
