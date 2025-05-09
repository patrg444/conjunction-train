#!/usr/bin/env python3
"""
This script downloads the UR-FUNNY V2 dataset and processes it into our
standard manifest format. The dataset contains ~3800 jokes from stand-up comedy
videos with manually annotated humor labels.

Reference: Hasan et al. (2019) - "UR-FUNNY: A Multimodal Language Dataset for Understanding Humor"
"""

import os
import pickle
import pandas as pd
import requests
import subprocess
import tempfile
import zipfile
import shutil
from tqdm import tqdm

# URL for the UR-FUNNY V2 features (pickle files)
FEATURES_URL = "https://www.dropbox.com/sh/9h0pcqmqoplx9p2/AAC8yYikSBVYCSFjm3afFHQva?dl=1"

def setup_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs("datasets/humor_datasets/ur_funny/v2", exist_ok=True)
    os.makedirs("datasets/manifests/humor", exist_ok=True)

def download_ur_funny_v2():
    """Download UR-FUNNY V2 dataset from Dropbox"""
    v2_dir = "datasets/humor_datasets/ur_funny/v2"
    
    # Check if key files already exist
    language_pkl = os.path.join(v2_dir, "language_sdk.pkl")
    humor_label_pkl = os.path.join(v2_dir, "humor_label_sdk.pkl")
    data_folds_pkl = os.path.join(v2_dir, "data_folds.pkl")
    
    if (os.path.exists(language_pkl) and 
        os.path.exists(humor_label_pkl) and 
        os.path.exists(data_folds_pkl)):
        print("UR-FUNNY V2 dataset files already exist. Skipping download.")
        return True
    
    print("Downloading UR-FUNNY V2 dataset from Dropbox...")
    print("This is a large download (~2.4GB) and may take some time...")
    
    # Create a temporary file to download the zip
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # First try with requests
        success = download_with_requests(FEATURES_URL, temp_path)
        
        # If requests fails, try with wget
        if not success:
            print("Attempting download with wget...")
            success = download_with_wget(FEATURES_URL, temp_path)
        
        # If wget fails, provide manual instructions
        if not success:
            print("\n====== DOWNLOAD FAILED ======")
            print("The automatic download failed. Please try to manually download the UR-FUNNY V2 dataset:")
            print("1. Visit: https://github.com/ROC-HCI/UR-FUNNY")
            print("2. Follow the links to download 'Pre-extracted multimodal features + labels (~2.4 GB)'")
            print("3. Unzip the files to 'datasets/humor_datasets/ur_funny/v2/'")
            print("4. Ensure you have the following files in that directory:")
            print("   - language_sdk.pkl")
            print("   - humor_label_sdk.pkl")
            print("   - data_folds.pkl")
            print("5. Run this script again after completing the manual download.")
            print("===============================\n")
            return False
        
        # Unzip the file
        print(f"Extracting files to {v2_dir}...")
        with zipfile.ZipFile(temp_path, 'r') as zip_ref:
            zip_ref.extractall(v2_dir)
        
        # Make sure the key files exist
        if (os.path.exists(language_pkl) and 
            os.path.exists(humor_label_pkl) and 
            os.path.exists(data_folds_pkl)):
            print("UR-FUNNY V2 dataset successfully downloaded and extracted.")
            return True
        else:
            # Check if the files are in a subdirectory
            subdirs = [d for d in os.listdir(v2_dir) if os.path.isdir(os.path.join(v2_dir, d))]
            for subdir in subdirs:
                subdir_path = os.path.join(v2_dir, subdir)
                files_in_subdir = os.listdir(subdir_path)
                if "language_sdk.pkl" in files_in_subdir:
                    # Move files up one level
                    for file in files_in_subdir:
                        shutil.move(os.path.join(subdir_path, file), v2_dir)
                    break
            
            # Check again if the key files exist
            if (os.path.exists(language_pkl) and 
                os.path.exists(humor_label_pkl) and 
                os.path.exists(data_folds_pkl)):
                print("UR-FUNNY V2 dataset successfully extracted from subdirectory.")
                return True
            else:
                print("Error: Could not find key pickle files after extraction.")
                return False
            
    except Exception as e:
        print(f"Error downloading or extracting UR-FUNNY V2 dataset: {e}")
        return False
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

def download_with_requests(url, output_path):
    """Download a file using the requests library"""
    try:
        print("Downloading with requests...")
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            print(f"Error: HTTP status code {response.status_code}")
            return False
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
        
        with open(output_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        
        progress_bar.close()
        return True
    except Exception as e:
        print(f"Error downloading with requests: {e}")
        return False

def download_with_wget(url, output_path):
    """Download a file using wget"""
    try:
        # Check if wget is available
        subprocess.run(['which', 'wget'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Download with wget
        cmd = ['wget', '--no-check-certificate', '-O', output_path, url]
        process = subprocess.run(cmd, check=True)
        
        return process.returncode == 0
    except subprocess.SubprocessError:
        print("wget command failed or not available")
        return False
    except Exception as e:
        print(f"Error downloading with wget: {e}")
        return False

def load_pickle_data(v2_dir):
    """Load pickle data from UR-FUNNY V2 dataset"""
    language_pkl = os.path.join(v2_dir, "language_sdk.pkl")
    humor_label_pkl = os.path.join(v2_dir, "humor_label_sdk.pkl")
    data_folds_pkl = os.path.join(v2_dir, "data_folds.pkl")
    
    try:
        print("Loading UR-FUNNY V2 dataset pickle files...")
        
        with open(language_pkl, 'rb') as f:
            language_data = pickle.load(f)
        
        with open(humor_label_pkl, 'rb') as f:
            humor_label_data = pickle.load(f)
        
        with open(data_folds_pkl, 'rb') as f:
            data_folds = pickle.load(f)
        
        return language_data, humor_label_data, data_folds
    except Exception as e:
        print(f"Error loading pickle data: {e}")
        return None, None, None

def process_ur_funny_to_manifest(language_data, humor_label_data, data_folds):
    """Process UR-FUNNY V2 dataset into manifest format"""
    if not language_data or not humor_label_data or not data_folds:
        print("Error: Dataset pickle files could not be loaded.")
        return {}
    
    train_ids = data_folds.get('train', [])
    dev_ids = data_folds.get('dev', [])
    
    print(f"Processing train split ({len(train_ids)} samples)...")
    train_data = []
    train_humor_count = 0
    train_non_humor_count = 0
    
    for video_id in tqdm(train_ids):
        # Skip if this ID is not in either dictionary
        if video_id not in language_data or video_id not in humor_label_data:
            continue
        
        # Get humor label
        label = int(humor_label_data[video_id])
        
        # Get punchline text
        punchline = ' '.join(language_data[video_id].get('punchline_sentence', []))
        
        # Get context sentences (optional)
        context_sentences = language_data[video_id].get('context_sentences', [])
        context_text = ""
        for context_sentence in context_sentences:
            if context_sentence:
                context_text += ' '.join(context_sentence) + ' '
        
        # Combine context and punchline with a separator
        full_text = context_text.strip()
        if full_text and punchline:
            full_text += " ||| "  # Separator between context and punchline
        full_text += punchline
        
        # Create manifest row
        manifest_row = {
            "talk_id": f"urfunny_{video_id}",
            "title": "UR-FUNNY",
            "text": full_text,
            "label": label
        }
        
        train_data.append(manifest_row)
        
        if label == 1:
            train_humor_count += 1
        else:
            train_non_humor_count += 1
    
    print(f"Processing validation split ({len(dev_ids)} samples)...")
    val_data = []
    val_humor_count = 0
    val_non_humor_count = 0
    
    for video_id in tqdm(dev_ids):
        # Skip if this ID is not in either dictionary
        if video_id not in language_data or video_id not in humor_label_data:
            continue
        
        # Get humor label
        label = int(humor_label_data[video_id])
        
        # Get punchline text
        punchline = ' '.join(language_data[video_id].get('punchline_sentence', []))
        
        # Get context sentences (optional)
        context_sentences = language_data[video_id].get('context_sentences', [])
        context_text = ""
        for context_sentence in context_sentences:
            if context_sentence:
                context_text += ' '.join(context_sentence) + ' '
        
        # Combine context and punchline with a separator
        full_text = context_text.strip()
        if full_text and punchline:
            full_text += " ||| "  # Separator between context and punchline
        full_text += punchline
        
        # Create manifest row
        manifest_row = {
            "talk_id": f"urfunny_{video_id}",
            "title": "UR-FUNNY",
            "text": full_text,
            "label": label
        }
        
        val_data.append(manifest_row)
        
        if label == 1:
            val_humor_count += 1
        else:
            val_non_humor_count += 1
    
    # Create DataFrames and save as CSV
    output_files = {}
    
    if train_data:
        train_output = "datasets/manifests/humor/ur_funny_train_humor.csv"
        train_df = pd.DataFrame(train_data)
        train_df.to_csv(train_output, index=False)
        output_files["train"] = train_output
        print(f"Created train manifest: {train_output}")
        print(f"  Total samples: {len(train_df)}")
        print(f"  Label 0 (non-humorous): {train_non_humor_count}")
        print(f"  Label 1 (humorous): {train_humor_count}")
    
    if val_data:
        val_output = "datasets/manifests/humor/ur_funny_val_humor.csv"
        val_df = pd.DataFrame(val_data)
        val_df.to_csv(val_output, index=False)
        output_files["val"] = val_output
        print(f"Created validation manifest: {val_output}")
        print(f"  Total samples: {len(val_df)}")
        print(f"  Label 0 (non-humorous): {val_non_humor_count}")
        print(f"  Label 1 (humorous): {val_humor_count}")
    
    return output_files

def main():
    """Main function to download and process UR-FUNNY V2 dataset"""
    setup_directories()
    
    # Download the dataset
    download_success = download_ur_funny_v2()
    if not download_success:
        print("Failed to download or extract UR-FUNNY V2 dataset.")
        return
    
    # Load the pickle files
    v2_dir = "datasets/humor_datasets/ur_funny/v2"
    language_data, humor_label_data, data_folds = load_pickle_data(v2_dir)
    
    # Process the data into manifests
    output_files = process_ur_funny_to_manifest(language_data, humor_label_data, data_folds)
    
    if output_files:
        print("\nUR-FUNNY V2 dataset processed successfully!")
        print("\nManifest files created:")
        for split, file in output_files.items():
            print(f"  {split}: {file}")
        
        print("\nYou can now train a model using these manifest files.")
        print("Example command:")
        print(f"  python enhanced_train_distil_humor.py --train_manifest {output_files.get('train', '')} --val_manifest {output_files.get('val', '')}")
    else:
        print("\nFailed to process UR-FUNNY V2 dataset.")

if __name__ == "__main__":
    main()
