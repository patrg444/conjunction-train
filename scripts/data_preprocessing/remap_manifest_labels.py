import csv
import os
from pathlib import Path
import argparse

# Mapping from RAVDESS filename emotion code to the desired 6-class index
# 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
# Target 6 classes: 0=angry, 1=disgust, 2=fear, 3=happy, 4=neutral, 5=sad
RAVDESS_CODE_TO_TARGET_IDX = {
    '01': 4,  # neutral -> neutral (4)
    '02': 4,  # calm -> neutral (4)
    '03': 3,  # happy -> happy (3)
    '04': 5,  # sad -> sad (5)
    '05': 0,  # angry -> angry (0)
    '06': 2,  # fearful -> fear (2)
    '07': 1,  # disgust -> disgust (1)
    '08': None, # surprised -> omit
}

def get_ravdess_emotion_code(filename):
    """Extracts the 3rd part (emotion code) from a RAVDESS filename."""
    parts = filename.split('-')
    if len(parts) == 7:
        return parts[2]
    return None

def is_ravdess_file(filepath_str):
    """Checks if a filepath likely belongs to the RAVDESS dataset based on naming."""
    filename = Path(filepath_str).name
    # Simple check: contains hyphens and ends with a common media extension
    return '-' in filename and filename.split('.')[-1].lower() in ['mp4', 'wav', 'flv', 'mp3']

def remap_manifest(input_path, output_path):
    """
    Reads the input manifest, remaps RAVDESS labels, and writes to output.
    - Maps RAVDESS calm (02) to neutral index (4).
    - Ensures RAVDESS neutral (01) is index 4.
    - Omits RAVDESS surprised (08).
    - Keeps other labels as they are.
    """
    lines_processed = 0
    lines_written = 0
    lines_omitted_surprised = 0
    lines_remapped_calm = 0
    lines_kept_other = 0
    ravdess_label_warnings = 0

    print(f"Processing manifest: {input_path}")
    print(f"Writing remapped manifest to: {output_path}")

    try:
        with open(input_path, 'r', newline='') as infile, \
             open(output_path, 'w', newline='') as outfile:

            # Use csv reader/writer to handle potential quoting/escaping issues, assuming TSV
            reader = csv.reader(infile, delimiter='\t')
            writer = csv.writer(outfile, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar=None, quotechar=None) # Write exactly as read if possible

            for row in reader:
                lines_processed += 1
                if len(row) != 2:
                    print(f"WARNING: Skipping malformed row {lines_processed}: {row}")
                    continue

                filepath_str, original_label_str = row
                new_label_str = original_label_str # Default to keeping original

                if is_ravdess_file(filepath_str):
                    filename = Path(filepath_str).name
                    emotion_code = get_ravdess_emotion_code(filename)

                    if emotion_code:
                        target_idx = RAVDESS_CODE_TO_TARGET_IDX.get(emotion_code)

                        if target_idx is None and emotion_code == '08': # Surprised
                            lines_omitted_surprised += 1
                            continue # Skip writing this line
                        elif target_idx is not None:
                            new_label_str = str(target_idx)
                            if emotion_code == '02' and original_label_str != new_label_str:
                                lines_remapped_calm += 1
                            # Optional: Check if original label for other emotions was already correct
                            elif original_label_str != new_label_str:
                                 print(f"WARNING: RAVDESS file {filename} (code {emotion_code}) had label {original_label_str}, expected {new_label_str} based on filename. Remapping.")
                                 ravdess_label_warnings += 1
                        else:
                             print(f"WARNING: Unknown RAVDESS emotion code '{emotion_code}' in filename {filename}. Keeping original label '{original_label_str}'.")
                             ravdess_label_warnings += 1
                             lines_kept_other += 1 # Count as kept
                    else:
                        print(f"WARNING: Could not parse RAVDESS emotion code from {filename}. Keeping original label '{original_label_str}'.")
                        ravdess_label_warnings += 1
                        lines_kept_other += 1 # Count as kept
                else:
                    # Assume non-RAVDESS files (like CREMA-D) have correct integer labels already
                    lines_kept_other += 1

                writer.writerow([filepath_str, new_label_str])
                lines_written += 1

    except FileNotFoundError:
        print(f"ERROR: Input manifest file not found: {input_path}")
        return False
    except Exception as e:
        print(f"ERROR: An error occurred during processing: {e}")
        return False

    print("\nRemapping Summary:")
    print(f"  Lines Processed: {lines_processed}")
    print(f"  Lines Written:   {lines_written}")
    print(f"  ----------------------------------")
    print(f"  RAVDESS Calm (02) remapped to Neutral (4): {lines_remapped_calm}")
    print(f"  RAVDESS Surprised (08) omitted:          {lines_omitted_surprised}")
    print(f"  Other lines kept/processed:              {lines_kept_other}")
    if ravdess_label_warnings > 0:
        print(f"  RAVDESS Label Warnings (kept original):    {ravdess_label_warnings}")

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remap emotion labels in a manifest file.")
    parser.add_argument("--input", type=str, default="data/audio_manifest.tsv", help="Path to the input manifest file.")
    parser.add_argument("--output", type=str, default="data/audio_manifest_remapped.tsv", help="Path to the output remapped manifest file.")
    args = parser.parse_args()

    if remap_manifest(args.input, args.output):
        print("\nRemapping successful.")
    else:
        print("\nRemapping failed.")
