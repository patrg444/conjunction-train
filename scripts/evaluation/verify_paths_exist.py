import argparse
import pandas as pd
from pathlib import Path
import os
import sys

def verify_paths(manifest_file, audio_base_path_str, video_base_path_str):
    """
    Verifies that audio and derived video paths from a manifest exist and stems match.
    """
    manifest_path = Path(manifest_file)
    audio_base_path = Path(audio_base_path_str)
    video_base_path = Path(video_base_path_str) # Assumed base for relative video paths

    print(f"Reading manifest: {manifest_path}")
    if not manifest_path.exists():
        print(f"Error: Manifest file not found at {manifest_path}")
        return False

    try:
        df = pd.read_csv(manifest_path)
    except Exception as e:
        print(f"Error reading manifest CSV: {e}")
        return False

    if 'path' not in df.columns:
        print("Error: Manifest CSV must contain a 'path' column (expected to contain relative audio paths).")
        return False

    total_rows = len(df)
    missing_audio_count = 0
    missing_video_count = 0
    stem_mismatch_count = 0
    errors_found = False
    max_errors_to_show = 10

    print(f"Verifying {total_rows} entries...")

    for index, row in df.iterrows():
        audio_relative_path_str = row['path']
        audio_relative_path = Path(audio_relative_path_str)
        
        # --- Construct Full Audio Path (Handle dataset-specific structures) ---
        if "ravdess/AudioWAV" in audio_relative_path_str:
            # RAVDESS audio is in datasets/ravdess_videos/Actor_XX/
            corrected_audio_relative_str = audio_relative_path_str.replace("ravdess/AudioWAV", "ravdess_videos", 1)
            audio_full_path = audio_base_path / corrected_audio_relative_str
        elif "crema_d/AudioWAV" in audio_relative_path_str:
             # Crema-D audio is in datasets/crema_d_videos/
             # Correctly replace 'crema_d/AudioWAV/' with 'crema_d_videos/'
             corrected_audio_relative_str = audio_relative_path_str.replace("crema_d/AudioWAV/", "crema_d_videos/", 1)
             audio_full_path = audio_base_path / corrected_audio_relative_str
        elif "crema_d/" in audio_relative_path_str: # Fallback if AudioWAV wasn't in path (unlikely based on manifest)
             corrected_audio_relative_str = audio_relative_path_str.replace("crema_d/", "crema_d_videos/", 1)
             audio_full_path = audio_base_path / corrected_audio_relative_str
        else:
             # Default assumption if not RAVDESS or Crema-D (might need adjustment)
             audio_full_path = audio_base_path / audio_relative_path

        # Check audio existence
        if not audio_full_path.exists():
            if missing_audio_count < max_errors_to_show:
                print(f"  [MISSING AUDIO] Row {index}: {audio_full_path}")
            missing_audio_count += 1
            errors_found = True
            # Continue to check video even if audio is missing, might reveal pattern
        
        # Derive and check video path
        video_found = False
        derived_video_full_path = None
        try:
            # Logic corrected based on actual file locations
            if "ravdess/" in audio_relative_path_str:
                # RAVDESS Video: Look for .mp4 in datasets/ravdess_videos/
                vid_rel_mp4 = audio_relative_path_str.replace("ravdess/AudioWAV", "ravdess_videos", 1).replace(".wav", ".mp4")
                vid_full_mp4 = video_base_path / vid_rel_mp4
                if vid_full_mp4.exists():
                    derived_video_full_path = vid_full_mp4
                else:
                     # RAVDESS video files are confirmed missing if not found here
                     if missing_video_count < max_errors_to_show:
                         print(f"  [MISSING VIDEO - RAVDESS] Row {index}: Audio={audio_full_path}, Searched MP4={vid_full_mp4}")
                     missing_video_count += 1
                     errors_found = True

            elif "crema_d/AudioWAV" in audio_relative_path_str:
                 # Crema-D Video: Look for .flv in datasets/crema_d_videos/
                 # Correctly replace 'crema_d/AudioWAV/' with 'crema_d_videos/' and change extension
                 vid_rel_flv_str = audio_relative_path_str.replace("crema_d/AudioWAV/", "crema_d_videos/", 1).replace(".wav", ".flv")
                 vid_full_flv = video_base_path / vid_rel_flv_str
                 if vid_full_flv.exists():
                     derived_video_full_path = vid_full_flv # Assign if found
                 # If not found, the missing video count happens below
            elif "crema_d/" in audio_relative_path_str: # Fallback if AudioWAV wasn't in path
                 vid_rel_flv_str = audio_relative_path_str.replace("crema_d/", "crema_d_videos/", 1).replace(".wav", ".flv")
                 vid_full_flv = video_base_path / vid_rel_flv_str
                 if vid_full_flv.exists():
                     derived_video_full_path = vid_full_flv # Assign if found
                 # If not found, the missing video count happens below

            # Moved the 'else' block for missing Crema-D videos outside the specific checks
            # This ensures it's only triggered if derived_video_full_path is still None after checks
            if derived_video_full_path is None and "crema_d" in audio_relative_path_str:
                 # Crema-D video files are confirmed missing if not found here
                 if missing_video_count < max_errors_to_show:
                     # Use the last calculated vid_full_flv for the error message
                     print(f"  [MISSING VIDEO - CREMA-D] Row {index}: Audio={audio_full_path}, Searched FLV={vid_full_flv}")
                 missing_video_count += 1
                 errors_found = True
            elif derived_video_full_path is None and "ravdess" not in audio_relative_path_str: # Handle default case missing video
                 # Use the last calculated vid_full_mp4 for the error message
                 if missing_video_count < max_errors_to_show:
                     print(f"  [MISSING VIDEO - DEFAULT] Row {index}: Audio={audio_full_path}, Searched MP4={vid_full_mp4}")
                 missing_video_count += 1
                 errors_found = True

            else: # Fallback/Default derivation (This block seems redundant now, consider removing if default case handled above)
                 vid_rel_mp4 = audio_relative_path_str.replace("AudioWAV", "VideoMP4", 1).replace(".wav", ".mp4")
                 vid_full_mp4 = video_base_path / vid_rel_mp4
                 if vid_full_mp4.exists():
                     derived_video_full_path = vid_full_mp4
                 else:
                     # This specific else block for default is now handled by the general check above
                     pass # No action needed here, handled by derived_video_full_path check

            # Stem check if video was found
            if derived_video_full_path and derived_video_full_path.exists():
                # Correctly indented block for stem check
                video_found = True # Mark video as found for logic purposes
                audio_stem = audio_relative_path.stem
                video_stem = derived_video_full_path.stem
                if audio_stem != video_stem:
                     if stem_mismatch_count < max_errors_to_show:
                         print(f"  [STEM MISMATCH] Row {index}: Audio Stem='{audio_stem}', Video Stem='{video_stem}' (Audio Path: {audio_full_path}, Video Path: {derived_video_full_path})")
                     stem_mismatch_count += 1
                     errors_found = True
            # Removed the 'elif not errors_found' block as missing video counting is handled within the derivation logic now


        except Exception as e_vid:
             if missing_video_count < max_errors_to_show:
                 print(f"  [ERROR DERIVING VIDEO] Row {index}: Audio={audio_full_path}, Error: {e_vid}")
             missing_video_count += 1 # Count as missing video due to error
             errors_found = True

    print("\n--- Verification Summary ---")
    print(f"Total rows checked: {total_rows}")
    print(f"Missing audio files: {missing_audio_count}")
    print(f"Missing video files (or derivation error): {missing_video_count}")
    print(f"Filename stem mismatches: {stem_mismatch_count}")

    if errors_found:
        print("\nVerification FAILED. Please check the errors above.")
        return False
    else:
        print("\nVerification PASSED. All files exist and stems match.")
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify audio/video file existence and stem matching based on a manifest.")
    parser.add_argument("manifest_file", help="Path to the manifest CSV file.")
    parser.add_argument("--audio_root", required=True, help="Root directory for audio files.")
    parser.add_argument("--video_root", required=True, help="Root directory for video files (where relative paths derived from manifest should resolve).")
    
    args = parser.parse_args()

    if verify_paths(args.manifest_file, args.audio_root, args.video_root):
        sys.exit(0) # Success
    else:
        sys.exit(1) # Failure
