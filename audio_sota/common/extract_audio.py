import os
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import soundfile as sf # Needed to check duration

def extract_audio_sox(input_file, output_wav_file, sample_rate=16000, channels=1, max_duration_sec=5.0):
    """Extracts/converts audio using sox, resamples, converts to mono, and truncates."""
    input_file = Path(input_file)
    output_wav_file = Path(output_wav_file)
    output_wav_file.parent.mkdir(parents=True, exist_ok=True)

    # Temporary file for initial conversion if input is not wav
    temp_wav = output_wav_file.with_suffix('.temp.wav') if input_file.suffix.lower() != '.wav' else output_wav_file

    # SoX command parts
    command = [
        'sox', str(input_file),        # Input file
        '-r', str(sample_rate),      # Resample to target rate
        '-c', str(channels),         # Convert to target channels (mono)
        '-b', '16',                  # Set bit depth to 16
        str(temp_wav),               # Output (potentially temporary) file
        # Removed gain -n, assuming normalization happens later if needed
    ]

    try:
        # Run SoX conversion/resampling
        result = subprocess.run(command, check=True, capture_output=True, text=True)

        # Check duration and truncate if needed (using soundfile)
        if max_duration_sec is not None and max_duration_sec > 0:
             info = sf.info(str(temp_wav))
             if info.duration > max_duration_sec:
                 print(f"Info: Truncating {temp_wav.name} from {info.duration:.2f}s to {max_duration_sec}s")
                 # Use another SoX command to truncate
                 truncate_command = [
                     'sox', str(temp_wav), str(output_wav_file), 'trim', '0', str(max_duration_sec)
                 ]
                 truncate_result = subprocess.run(truncate_command, check=True, capture_output=True, text=True)
                 if temp_wav != output_wav_file: # Clean up temp file if it was used
                      temp_wav.unlink()
                 return True, None # Success after truncation
             
        # If no truncation needed and temp file was used, rename/move it
        if temp_wav != output_wav_file:
             temp_wav.rename(output_wav_file)
             
        return True, None # Success without truncation

    except subprocess.CalledProcessError as e:
        error_message = f"SoX error for {input_file.name}:\nSTDERR: {e.stderr}\nSTDOUT: {e.stdout}"
        # Clean up temp file on error if it exists
        if temp_wav.exists() and temp_wav != output_wav_file:
             try:
                 temp_wav.unlink()
             except OSError:
                 pass # Ignore cleanup error
        return False, error_message
    except FileNotFoundError:
         # Clean up temp file on error if it exists
         if temp_wav.exists() and temp_wav != output_wav_file:
              try:
                  temp_wav.unlink()
              except OSError:
                  pass # Ignore cleanup error
         return False, "SoX command not found. Is sox installed and in your system's PATH?"
    except Exception as e: # Catch other potential errors like soundfile issues
         # Clean up temp file on error if it exists
         if temp_wav.exists() and temp_wav != output_wav_file:
              try:
                  temp_wav.unlink()
              except OSError:
                  pass # Ignore cleanup error
         return False, f"Error processing {input_file.name}: {e}"


def process_file(args):
    """Wrapper function for concurrent processing using SoX."""
    input_file, raw_input_dir, output_audio_dir, max_duration = args
    relative_path = input_file.relative_to(raw_input_dir)
    output_wav_path = output_audio_dir / relative_path.with_suffix('.wav')
    
    # Skip if already exists
    if output_wav_path.exists():
        return input_file.name, "skipped"

    # Use the new SoX function
    success, error = extract_audio_sox(input_file, output_wav_path, max_duration_sec=max_duration) 
    if success:
        return input_file.name, "success"
    else:
        return input_file.name, f"failed: {error}"


def extract_all_audio(dataset_name, raw_input_dir, output_audio_dir, max_duration_sec=5.0, max_workers=None):
    """Finds all relevant media files (video or audio) and extracts/converts audio using SoX in parallel."""
    raw_path = Path(raw_input_dir)
    output_path = Path(output_audio_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning {raw_path} for media files ({dataset_name})...")
    
    # Find relevant input files based on dataset
    if dataset_name == 'crema_d':
        # CREMA-D original is video (.flv), but might already be extracted to .wav
        # Prioritize existing WAVs if they are in the input dir, otherwise look for FLV
        input_files = list(raw_path.rglob('*.wav')) 
        if not input_files:
             print(f"No .wav found in {raw_path}, looking for .flv video files...")
             input_files = list(raw_path.rglob('*.flv'))
             if not input_files:
                  print(f"Error: No .wav or .flv files found in {raw_path}")
                  return
        print(f"Found {len(input_files)} input files for CREMA-D.")

    elif dataset_name == 'ravdess':
         # RAVDESS original is video (.mp4), but might already be extracted to .wav
         # Prioritize existing WAVs if they are in the input dir, otherwise look for MP4
         input_files = []
         actor_dirs = [d for d in raw_path.iterdir() if d.is_dir() and d.name.startswith('Actor_')]
         if not actor_dirs:
              print(f"Error: No Actor_* subdirectories found in {raw_path}")
              return
         for actor_dir in actor_dirs:
             input_files.extend(list(actor_dir.rglob('*.wav'))) # Check for WAV first
         
         if not input_files:
              print(f"No .wav found in Actor subdirs of {raw_path}, looking for .mp4 video files...")
              for actor_dir in actor_dirs:
                  input_files.extend(list(actor_dir.rglob('*.mp4'))) # Fallback to MP4
              if not input_files:
                   print(f"Error: No .wav or .mp4 files found in Actor subdirectories of {raw_path}")
                   return
         print(f"Found {len(input_files)} input files for RAVDESS.")
    
    elif dataset_name == 'ur_funny':
        # UR-FUNNY videos are provided as .mp4 files
        input_files = list(raw_path.rglob('*.mp4'))
        if not input_files:
            print(f"Error: No .mp4 files found in {raw_path} for UR-FUNNY.")
            return
        print(f"Found {len(input_files)} input files for UR-FUNNY.")

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    print(f"Found {len(input_files)} total input files. Starting audio conversion/extraction...")

    tasks = [(infile, raw_path, output_path, max_duration_sec) for infile in input_files]
    success_count = 0
    skipped_count = 0
    failed_count = 0
    
    # Use ThreadPoolExecutor for I/O bound tasks (sox process)
    # Adjust max_workers based on your system's core count for optimal performance
    if max_workers is None:
        max_workers = os.cpu_count() or 4 # Default to 4 if cpu_count fails

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_file, tasks), total=len(tasks), desc=f"Processing {dataset_name} audio with SoX"))

    print("\nProcessing Summary:")
    for filename, status in results:
        if status == "success":
            success_count += 1
        elif status == "skipped":
             skipped_count += 1
        else:
            failed_count += 1
            print(f"- {filename}: {status}") # Print errors for failed files

    print(f"\nSuccessfully processed/converted: {success_count}")
    print(f"Skipped (already exist): {skipped_count}")
    print(f"Failed: {failed_count}")
    print(f"\nOutput WAV files saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract/convert audio from datasets using SoX, resample, and truncate.")
    # The following line was already corrected in the previous successful edit, 
    # but the error message showed the reverted state. Re-applying the change based on the error message content.
    parser.add_argument('--dataset', required=True, choices=['crema_d', 'ravdess', 'ur_funny'], help='Name of the dataset.') 
    parser.add_argument('--input_dir', required=True, help='Path to the directory containing raw media files (e.g., FLV/MP4 or existing WAVs).')
    parser.add_argument('--output_dir', required=True, help='Path to the directory to save the processed 16kHz mono WAV files.')
    parser.add_argument('--max_duration', type=float, default=5.0, help='Maximum duration in seconds to keep (clips longer than this will be truncated). Set to 0 or negative to disable truncation.')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers (default: CPU count).')

    args = parser.parse_args()

    extract_all_audio(
        args.dataset,
        args.input_dir,
        args.output_dir,
        max_duration_sec=args.max_duration,
        max_workers=args.workers
    )
