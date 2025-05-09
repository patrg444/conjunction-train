import pandas as pd
import os
import argparse
from tqdm import tqdm

def generate_multimodal_manifest(input_manifest_path, embedding_base_dir, output_manifest_path):
    """
    Generates a multimodal manifest CSV file by combining an input manifest
    with paths to text, audio, and video embeddings.

    Args:
        input_manifest_path (str): Path to the input CSV manifest (needs 'file_id', 'label').
        embedding_base_dir (str): Base directory where modality subdirectories ('text', 'audio', 'video') exist.
        output_manifest_path (str): Path to save the generated multimodal manifest CSV.
    """
    print(f"Reading input manifest from: {input_manifest_path}")
    try:
        input_df = pd.read_csv(input_manifest_path)
    except FileNotFoundError:
        print(f"Error: Input manifest file not found at {input_manifest_path}")
        return
    except Exception as e:
        print(f"Error reading input manifest: {e}")
        return

    # Use 'talk_id' as the identifier column
    id_column = 'talk_id'
    if id_column not in input_df.columns or 'label' not in input_df.columns:
        print(f"Error: Input manifest must contain '{id_column}' and 'label' columns.")
        return

    print(f"Using embedding base directory: {embedding_base_dir}")
    output_data = []
    missing_files_count = 0
    processed_count = 0

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_manifest_path), exist_ok=True)

    print("Processing entries and verifying embedding files...")
    for _, row in tqdm(input_df.iterrows(), total=input_df.shape[0], desc="Generating Manifest"):
        talk_id = row[id_column] # Read from the correct ID column
        label = row['label']

        text_emb_path = os.path.join(embedding_base_dir, 'text', f"{talk_id}.npy")
        audio_emb_path = os.path.join(embedding_base_dir, 'audio', f"{talk_id}.npy")
        video_emb_path = os.path.join(embedding_base_dir, 'video', f"{talk_id}.npy")

        # Verify all embedding files exist
        text_exists = os.path.exists(text_emb_path)
        audio_exists = os.path.exists(audio_emb_path)
        video_exists = os.path.exists(video_emb_path)

        if text_exists and audio_exists and video_exists:
            output_data.append({
                id_column: talk_id, # Write to the correct ID column
                'text_embedding_path': text_emb_path,
                'audio_embedding_path': audio_emb_path,
                'video_embedding_path': video_emb_path,
                'label': label
            })
            processed_count += 1
        else:
            missing_files_count += 1
            print(f"Warning: Missing embeddings for {id_column} '{talk_id}': "
                  f"Text={text_exists}, Audio={audio_exists}, Video={video_exists}")

    if not output_data:
        print("Error: No valid entries found with all required embeddings. Output manifest will be empty.")
        return

    print(f"\nProcessed {processed_count} entries.")
    if missing_files_count > 0:
        print(f"Warning: Skipped {missing_files_count} entries due to missing embedding files.")

    output_df = pd.DataFrame(output_data)

    print(f"Saving multimodal manifest to: {output_manifest_path}")
    try:
        output_df.to_csv(output_manifest_path, index=False)
        print("Multimodal manifest generated successfully.")
    except Exception as e:
        print(f"Error writing output manifest: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a multimodal manifest from embeddings.")
    parser.add_argument("--input_manifest", required=True, help="Path to the input CSV manifest (e.g., ur_funny_all_humor_raw.csv).")
    parser.add_argument("--embedding_dir", required=True, help="Base directory containing 'text', 'audio', 'video' embedding subdirectories.")
    parser.add_argument("--output_manifest", required=True, help="Path to save the generated multimodal manifest CSV.")

    args = parser.parse_args()

    generate_multimodal_manifest(args.input_manifest, args.embedding_dir, args.output_manifest)
