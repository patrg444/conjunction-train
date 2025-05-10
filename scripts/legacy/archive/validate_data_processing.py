#!/usr/bin/env python3
"""
Validation script to verify the correctness of data segmentation and labeling.
This script performs several checks:
1. Verifies RAVDESS filename parsing is correct
2. Validates emotion labels in processed data match the expected emotions from filenames
3. Visualizes segmented data with labels for manual inspection
4. Analyzes the distribution of emotions and segment lengths for consistency
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import argparse
import random

# RAVDESS emotion mapping
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# Function to parse RAVDESS filename
def parse_ravdess_filename(filename):
    """Parse RAVDESS filename to extract metadata.
    
    Format: XX-XX-XX-XX-XX-XX-XX.[wav/mp4]
    Positions:
    1. Modality (01=full-AV, 02=video-only, 03=audio-only)
    2. Channel (01=speech, 02=song)
    3. Emotion (01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised)
    4. Emotional intensity (01=normal, 02=strong)
    5. Statement (01="Kids are talking by the door", 02="Dogs are sitting by the door")
    6. Repetition (01=1st repetition, 02=2nd repetition)
    7. Actor (01 to 24)
    """
    try:
        basename = os.path.basename(filename)
        parts = basename.split('-')
        
        if len(parts) != 7:
            if len(parts) == 8:  # Some files may have an extra part
                parts = parts[:7]
            else:
                return None
        
        # Extract components
        modality = parts[0]
        channel = parts[1]
        emotion = parts[2]
        intensity = parts[3]
        statement = parts[4]
        repetition = parts[5]
        actor = parts[6].split('.')[0]  # Remove file extension
        
        # Map emotion code to name
        emotion_name = EMOTION_MAP.get(emotion, f"unknown-{emotion}")
        
        # Handle numeric emotion for model input (0-7 for 8 emotions)
        numeric_emotion = int(emotion) - 1 if emotion in EMOTION_MAP else -1
        
        return {
            "modality": modality,
            "channel": channel,
            "emotion": emotion,
            "emotion_name": emotion_name,
            "numeric": numeric_emotion,
            "intensity": intensity,
            "statement": statement,
            "repetition": repetition,
            "actor": actor,
            "filename": basename
        }
    except Exception as e:
        print(f"Error parsing filename {filename}: {str(e)}")
        return None

def validate_filename_parsing(npz_files, verbose=True):
    """Validate that the RAVDESS filenames are being parsed correctly."""
    if verbose:
        print("\n===== VALIDATING FILENAME PARSING =====")
    
    valid_count = 0
    invalid_count = 0
    
    # Take a sample of files for detailed output if there are many
    sample_size = min(10, len(npz_files))
    sample_files = random.sample(npz_files, sample_size) if len(npz_files) > sample_size else npz_files
    
    for file_path in npz_files:
        basename = os.path.basename(file_path)
        metadata = parse_ravdess_filename(basename)
        
        if metadata is None:
            invalid_count += 1
            if file_path in sample_files or verbose:
                print(f"❌ Failed to parse: {basename}")
        else:
            valid_count += 1
            if file_path in sample_files and verbose:
                print(f"✅ {basename} → Emotion: {metadata['emotion_name']} ({metadata['numeric']}), Actor: {metadata['actor']}")
    
    success_rate = (valid_count / len(npz_files)) * 100 if npz_files else 0
    
    if verbose:
        print(f"\nParsed {valid_count}/{len(npz_files)} filenames successfully ({success_rate:.1f}%)")
        if invalid_count > 0:
            print(f"⚠️ {invalid_count} files could not be parsed properly")
    
    return success_rate >= 95  # Consider successful if at least 95% parse correctly

def validate_emotion_labels(npz_files, verbose=True):
    """Validate that the emotion labels in the processed data match the expected emotions from filenames."""
    if verbose:
        print("\n===== VALIDATING EMOTION LABELS =====")
    
    correct_labels = 0
    incorrect_labels = 0
    unlabeled_files = 0
    
    # Take a sample of files for detailed output
    sample_size = min(10, len(npz_files))
    sample_files = random.sample(npz_files, sample_size) if len(npz_files) > sample_size else npz_files
    
    for file_path in npz_files:
        basename = os.path.basename(file_path)
        metadata = parse_ravdess_filename(basename)
        
        if metadata is None:
            continue
        
        try:
            data = np.load(file_path, allow_pickle=True)
            
            if 'emotion_label' not in data:
                unlabeled_files += 1
                if file_path in sample_files or verbose:
                    print(f"⚠️ {basename}: No emotion_label found in NPZ file")
                continue
            
            emotion_label = data['emotion_label']
            
            # If it's a scalar, convert to an integer if needed
            if np.isscalar(emotion_label) or (isinstance(emotion_label, np.ndarray) and emotion_label.size == 1):
                emotion_label = emotion_label.item() if isinstance(emotion_label, np.ndarray) else emotion_label
                
                # Compare with expected emotion from filename
                if emotion_label == metadata['numeric']:
                    correct_labels += 1
                    if file_path in sample_files and verbose:
                        print(f"✅ {basename}: Label {emotion_label} matches expected {metadata['numeric']} ({metadata['emotion_name']})")
                else:
                    incorrect_labels += 1
                    if file_path in sample_files or verbose:
                        print(f"❌ {basename}: Label {emotion_label} does NOT match expected {metadata['numeric']} ({metadata['emotion_name']})")
            else:
                # Handle case where there are multiple labels (e.g., per sequence)
                # We'll just check that all labels are consistent
                unique_labels = set(emotion_label) if hasattr(emotion_label, '__iter__') else {emotion_label}
                
                if len(unique_labels) == 1 and metadata['numeric'] in unique_labels:
                    correct_labels += 1
                    if file_path in sample_files and verbose:
                        print(f"✅ {basename}: All {len(emotion_label)} labels match expected {metadata['numeric']} ({metadata['emotion_name']})")
                else:
                    incorrect_labels += 1
                    if file_path in sample_files or verbose:
                        print(f"❌ {basename}: Labels {unique_labels} do NOT match expected {metadata['numeric']} ({metadata['emotion_name']})")
        
        except Exception as e:
            print(f"Error checking labels in {file_path}: {str(e)}")
    
    total_checked = correct_labels + incorrect_labels
    success_rate = (correct_labels / total_checked) * 100 if total_checked > 0 else 0
    
    if verbose:
        print(f"\nVerified {correct_labels}/{total_checked} labels correctly ({success_rate:.1f}%)")
        if incorrect_labels > 0:
            print(f"⚠️ {incorrect_labels} files had incorrect labels")
        if unlabeled_files > 0:
            print(f"⚠️ {unlabeled_files} files had no emotion labels")
    
    return success_rate >= 95  # Consider successful if at least 95% correct

def visualize_segmentation(npz_files, num_samples=3, output_dir="validation_visualizations"):
    """Visualize segmented data with labels for manual inspection."""
    print("\n===== VISUALIZING SEGMENTED DATA =====")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Randomly sample files to visualize
    if len(npz_files) > num_samples:
        files_to_visualize = random.sample(npz_files, num_samples)
    else:
        files_to_visualize = npz_files
    
    for file_idx, file_path in enumerate(files_to_visualize):
        basename = os.path.basename(file_path)
        metadata = parse_ravdess_filename(basename)
        
        if metadata is None:
            print(f"⚠️ Skipping visualization for {basename}: Could not parse filename")
            continue
        
        try:
            data = np.load(file_path, allow_pickle=True)
            
            # Check if we have the necessary data
            if 'video_sequences' not in data or 'audio_sequences' not in data:
                print(f"⚠️ Skipping visualization for {basename}: Missing video or audio sequences")
                continue
            
            video_seqs = data['video_sequences']
            audio_seqs = data['audio_sequences']
            
            if len(video_seqs) == 0 or len(audio_seqs) == 0:
                print(f"⚠️ Skipping visualization for {basename}: Empty sequences")
                continue
            
            # Get emotion label
            emotion_name = metadata['emotion_name']
            if 'emotion_label' in data:
                emotion_label = data['emotion_label']
                if np.isscalar(emotion_label) or (isinstance(emotion_label, np.ndarray) and emotion_label.size == 1):
                    emotion_label = emotion_label.item() if isinstance(emotion_label, np.ndarray) else emotion_label
                    emotion_name = EMOTION_MAP.get(f"{emotion_label+1:02d}", f"emotion_{emotion_label}")
            
            # Create a multi-part figure for this file
            fig = plt.figure(figsize=(15, 5 * min(3, len(video_seqs))))
            fig.suptitle(f"Segmentation for {basename}\nEmotion: {emotion_name}", fontsize=16)
            
            # Determine number of segments to show (max 3)
            num_segments = min(3, len(video_seqs))
            
            for i in range(num_segments):
                video_seq = video_seqs[i]
                audio_seq = audio_seqs[i]
                
                # Create a 2-row subplot for video and audio features
                ax1 = plt.subplot(num_segments, 2, i*2 + 1)
                ax2 = plt.subplot(num_segments, 2, i*2 + 2)
                
                # Plot video features (first 100 dimensions for visibility)
                max_dims_to_show = min(100, video_seq.shape[1])
                video_data = video_seq[:, :max_dims_to_show]
                im1 = ax1.imshow(video_data.T, aspect='auto', cmap='viridis')
                ax1.set_title(f"Video Features - Segment {i+1}/{len(video_seqs)}")
                ax1.set_xlabel("Time Steps")
                ax1.set_ylabel("Feature Dimensions")
                plt.colorbar(im1, ax=ax1)
                
                # Plot audio features
                max_dims_to_show = min(100, audio_seq.shape[1])
                audio_data = audio_seq[:, :max_dims_to_show]
                im2 = ax2.imshow(audio_data.T, aspect='auto', cmap='plasma')
                ax2.set_title(f"Audio Features - Segment {i+1}/{len(audio_seqs)}")
                ax2.set_xlabel("Time Steps")
                ax2.set_ylabel("Feature Dimensions")
                plt.colorbar(im2, ax=ax2)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
            
            # Save figure
            output_file = os.path.join(output_dir, f"segmentation_vis_{file_idx+1}_{os.path.splitext(basename)[0]}.png")
            plt.savefig(output_file, dpi=150)
            plt.close()
            
            print(f"✅ Created visualization: {output_file}")
            
        except Exception as e:
            print(f"Error visualizing {file_path}: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    print(f"\nVisualizations saved to {output_dir}")
    return True

def analyze_distributions(npz_files, output_dir="validation_visualizations"):
    """Analyze the distribution of emotions, segment lengths, and other properties."""
    print("\n===== ANALYZING DATA DISTRIBUTIONS =====")
    
    os.makedirs(output_dir, exist_ok=True)
    
    emotion_counts = Counter()
    sequence_counts = []
    video_seq_lengths = []
    audio_seq_lengths = []
    video_dim_sizes = []
    audio_dim_sizes = []
    
    for file_path in npz_files:
        basename = os.path.basename(file_path)
        metadata = parse_ravdess_filename(basename)
        
        if metadata is None:
            continue
        
        # Count emotions
        emotion_counts[metadata['emotion_name']] += 1
        
        try:
            data = np.load(file_path, allow_pickle=True)
            
            # Count sequences per file
            if 'video_sequences' in data:
                video_seqs = data['video_sequences']
                sequence_counts.append(len(video_seqs))
                
                # Collect sequence lengths and dimensions
                for seq in video_seqs:
                    video_seq_lengths.append(seq.shape[0])
                    video_dim_sizes.append(seq.shape[1])
            
            if 'audio_sequences' in data:
                audio_seqs = data['audio_sequences']
                
                # Collect sequence lengths and dimensions
                for seq in audio_seqs:
                    audio_seq_lengths.append(seq.shape[0])
                    audio_dim_sizes.append(seq.shape[1])
                    
        except Exception as e:
            print(f"Error analyzing {file_path}: {str(e)}")
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # 1. Emotion distribution
    plt.subplot(2, 2, 1)
    emotions = list(emotion_counts.keys())
    counts = [emotion_counts[e] for e in emotions]
    plt.bar(emotions, counts)
    plt.title('Emotion Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Number of Files')
    
    # 2. Sequence count distribution
    plt.subplot(2, 2, 2)
    plt.hist(sequence_counts, bins=range(1, max(sequence_counts) + 2), align='left', alpha=0.7)
    plt.title('Sequences per File Distribution')
    plt.xlabel('Number of Sequences')
    plt.ylabel('Number of Files')
    plt.xticks(range(1, max(sequence_counts) + 1))
    
    # 3. Video sequence length distribution
    plt.subplot(2, 2, 3)
    plt.hist(video_seq_lengths, bins=10, alpha=0.7)
    plt.title('Video Sequence Length Distribution')
    plt.xlabel('Sequence Length (Time Steps)')
    plt.ylabel('Count')
    
    # 4. Audio sequence length distribution
    plt.subplot(2, 2, 4)
    plt.hist(audio_seq_lengths, bins=10, alpha=0.7)
    plt.title('Audio Sequence Length Distribution')
    plt.xlabel('Sequence Length (Time Steps)')
    plt.ylabel('Count')
    
    plt.tight_layout()
    
    # Save the distributions
    output_file = os.path.join(output_dir, "data_distributions.png")
    plt.savefig(output_file, dpi=150)
    plt.close()
    
    # Print summary statistics
    print(f"Total files analyzed: {len(npz_files)}")
    print("\nEmotion Distribution:")
    for emotion, count in emotion_counts.most_common():
        print(f"  {emotion}: {count} files ({count/len(npz_files)*100:.1f}%)")
    
    print("\nSequence Statistics:")
    print(f"  Average sequences per file: {np.mean(sequence_counts):.1f}")
    print(f"  Min/Max sequences per file: {min(sequence_counts)}/{max(sequence_counts)}")
    
    print("\nSequence Length Statistics:")
    print(f"  Video sequences: {np.mean(video_seq_lengths):.1f} time steps on average (std: {np.std(video_seq_lengths):.1f})")
    print(f"  Audio sequences: {np.mean(audio_seq_lengths):.1f} time steps on average (std: {np.std(audio_seq_lengths):.1f})")
    
    print(f"\nDistribution visualizations saved to {output_file}")
    return True

def validate_data_consistency(npz_files, verbose=True):
    """Validate the consistency of the processed data (sequence counts, dimensions, etc.)."""
    if verbose:
        print("\n===== VALIDATING DATA CONSISTENCY =====")
    
    consistent_files = 0
    inconsistent_files = 0
    
    for file_path in npz_files:
        basename = os.path.basename(file_path)
        
        try:
            data = np.load(file_path, allow_pickle=True)
            
            issues = []
            
            # Check that we have both video and audio sequences
            if 'video_sequences' not in data:
                issues.append("Missing video_sequences")
            if 'audio_sequences' not in data:
                issues.append("Missing audio_sequences")
            
            if len(issues) > 0:
                inconsistent_files += 1
                if verbose:
                    print(f"❌ {basename}: {', '.join(issues)}")
                continue
            
            video_seqs = data['video_sequences']
            audio_seqs = data['audio_sequences']
            
            # Check that we have equal number of video and audio sequences
            if len(video_seqs) != len(audio_seqs):
                inconsistent_files += 1
                if verbose:
                    print(f"❌ {basename}: Mismatched sequence counts - video: {len(video_seqs)}, audio: {len(audio_seqs)}")
                continue
            
            # Check that all sequences are not empty
            if len(video_seqs) == 0 or len(audio_seqs) == 0:
                inconsistent_files += 1
                if verbose:
                    print(f"❌ {basename}: Empty sequences")
                continue
            
            # Check that video and audio sequences have consistent dimensions
            video_dims = set(seq.shape[1] for seq in video_seqs)
            audio_dims = set(seq.shape[1] for seq in audio_seqs)
            
            if len(video_dims) > 1:
                issues.append(f"Inconsistent video dimensions: {video_dims}")
            if len(audio_dims) > 1:
                issues.append(f"Inconsistent audio dimensions: {audio_dims}")
            
            if len(issues) > 0:
                inconsistent_files += 1
                if verbose:
                    print(f"❌ {basename}: {', '.join(issues)}")
                continue
            
            # All checks passed
            consistent_files += 1
            if verbose and random.random() < 0.1:  # Only show ~10% of successful files to reduce output
                print(f"✅ {basename}: {len(video_seqs)} sequences with consistent dimensions")
            
        except Exception as e:
            inconsistent_files += 1
            if verbose:
                print(f"❌ {basename}: Error loading/analyzing file - {str(e)}")
    
    total_files = consistent_files + inconsistent_files
    success_rate = (consistent_files / total_files) * 100 if total_files > 0 else 0
    
    if verbose:
        print(f"\nData consistency: {consistent_files}/{total_files} files consistent ({success_rate:.1f}%)")
        if inconsistent_files > 0:
            print(f"⚠️ {inconsistent_files} files had consistency issues")
    
    return success_rate >= 90  # Consider successful if at least 90% consistent

def main():
    """Main function for validating the processed dataset."""
    parser = argparse.ArgumentParser(description='Validate RAVDESS data processing and segmentation.')
    parser.add_argument('--data-dir', type=str, default='processed_features_large',
                        help='Directory containing processed features (default: processed_features_large)')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualizations of segmented data')
    parser.add_argument('--num-samples', type=int, default=3,
                        help='Number of files to visualize (if --visualize is used)')
    parser.add_argument('--output-dir', type=str, default='validation_visualizations',
                        help='Directory to save visualizations (default: validation_visualizations)')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce verbosity of output')
    
    args = parser.parse_args()
    
    # Find all .npz files in the specified directory
    file_pattern = os.path.join(args.data_dir, "*.npz")
    npz_files = glob.glob(file_pattern)
    
    if not npz_files:
        print(f"No .npz files found in {args.data_dir}")
        return False
    
    print(f"Found {len(npz_files)} .npz files in {args.data_dir}")
    
    # Run validation checks
    validation_results = {}
    
    # 1. Validate filename parsing
    validation_results['filename_parsing'] = validate_filename_parsing(npz_files, verbose=not args.quiet)
    
    # 2. Validate emotion labels
    validation_results['emotion_labels'] = validate_emotion_labels(npz_files, verbose=not args.quiet)
    
    # 3. Validate data consistency
    validation_results['data_consistency'] = validate_data_consistency(npz_files, verbose=not args.quiet)
    
    # 4. Analyze distributions
    analyze_distributions(npz_files, output_dir=args.output_dir)
    
    # 5. Visualize segmentation if requested
    if args.visualize:
        visualize_segmentation(npz_files, num_samples=args.num_samples, output_dir=args.output_dir)
    
    # Print summary of validation results
    print("\n===== VALIDATION SUMMARY =====")
    all_passed = True
    for check, passed in validation_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {check.replace('_', ' ').title()}")
        all_passed = all_passed and passed
    
    if all_passed:
        print("\n✅ All validation checks passed! The data segmentation and labeling process appears to be correct.")
    else:
        print("\n⚠️ Some validation checks failed. Review the issues above and consider reprocessing the data.")
    
    return all_passed

if __name__ == "__main__":
    main()
