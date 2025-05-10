#!/usr/bin/env python3
"""
Process RAVDESS dataset to extract audio and video features with OpenSMILE.
This script processes videos from the RAVDESS dataset until at least 250 segments
are generated with 1-second window size and 0.5-second hop size.

Enhanced features:
- Command-line argument parsing for more flexibility
- Option to process specific emotion categories or actors
- Functionality to balance the dataset across emotion categories
- Visualization of processed features
- Option to validate the processed features
- Fixed multiprocessing resource leak
"""

import os
import sys
import glob
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, set_start_method
from multimodal_preprocess import process_video_for_multimodal_lstm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("process_ravdess.log"),
        logging.StreamHandler()
    ]
)

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

# RAVDESS intensity mapping
INTENSITY_MAP = {
    "01": "normal",
    "02": "strong"
}

def visualize_features(npz_file, output_dir=None):
    """Visualize features from a processed NPZ file.
    
    Args:
        npz_file: Path to NPZ file with processed features
        output_dir: Directory to save visualization. If None, only shows plot
    
    Returns:
        Path to saved visualization file if output_dir is provided, else None
    """
    try:
        # Load the data
        data = np.load(npz_file, allow_pickle=True)
        
        # Extract sequences
        video_sequences = data['video_sequences']
        audio_sequences = data['audio_sequences']
        
        if isinstance(video_sequences, np.ndarray) and video_sequences.dtype == np.dtype('O'):
            # Handle object arrays
            if len(video_sequences) == 0:
                logging.error(f"No video sequences found in {npz_file}")
                return None
            
            # Use the first sequence for visualization
            video_seq = video_sequences[0]
            audio_seq = audio_sequences[0]
        else:
            # Direct arrays
            video_seq = video_sequences
            audio_seq = audio_sequences
        
        # Create visualization
        fig, axs = plt.subplots(2, 1, figsize=(12, 8))
        
        # Video features heatmap (using first 100 dimensions for visibility if too large)
        max_dims_to_show = 100
        video_data = video_seq[:, :max_dims_to_show] if video_seq.shape[1] > max_dims_to_show else video_seq
        im1 = axs[0].imshow(video_data.T, aspect='auto', cmap='viridis')
        axs[0].set_title(f'Video Features (showing {video_data.shape[1]} of {video_seq.shape[1]} dimensions)')
        axs[0].set_xlabel('Time Steps')
        axs[0].set_ylabel('Feature Dimensions')
        plt.colorbar(im1, ax=axs[0])
        
        # Audio features heatmap
        max_dims_to_show = 100
        audio_data = audio_seq[:, :max_dims_to_show] if audio_seq.shape[1] > max_dims_to_show else audio_seq
        im2 = axs[1].imshow(audio_data.T, aspect='auto', cmap='plasma')
        axs[1].set_title(f'Audio Features (showing {audio_data.shape[1]} of {audio_seq.shape[1]} dimensions)')
        axs[1].set_xlabel('Time Steps')
        axs[1].set_ylabel('Feature Dimensions')
        plt.colorbar(im2, ax=axs[1])
        
        # Add overall title
        plt.suptitle(f'Features from {os.path.basename(npz_file)}', fontsize=16)
        plt.tight_layout()
        
        # Save or show
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(npz_file))[0]}_visualization.png")
            plt.savefig(output_file, dpi=150)
            plt.close()
            logging.info(f"Saved visualization to {output_file}")
            return output_file
        else:
            plt.show()
            return None
            
    except Exception as e:
        logging.error(f"Error visualizing features from {npz_file}: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None

def validate_processed_features(npz_file):
    """Validate processed features from an NPZ file.
    
    Args:
        npz_file: Path to NPZ file with processed features
        
    Returns:
        Boolean indicating whether the file is valid
    """
    try:
        # Load the data
        data = np.load(npz_file, allow_pickle=True)
        
        # Check required keys
        required_keys = ['video_sequences', 'audio_sequences', 'window_start_times']
        for key in required_keys:
            if key not in data:
                logging.error(f"Missing required key {key} in {npz_file}")
                return False
        
        # Check video sequences
        video_sequences = data['video_sequences']
        if len(video_sequences) == 0:
            logging.error(f"Empty video sequences in {npz_file}")
            return False
            
        # Check audio sequences
        audio_sequences = data['audio_sequences']
        if len(audio_sequences) == 0:
            logging.error(f"Empty audio sequences in {npz_file}")
            return False
            
        # Check matching sequence counts
        if len(video_sequences) != len(audio_sequences):
            logging.error(f"Mismatched sequence counts in {npz_file}: video={len(video_sequences)}, audio={len(audio_sequences)}")
            return False
            
        # Everything looks good
        return True
        
    except Exception as e:
        logging.error(f"Error validating {npz_file}: {str(e)}")
        return False

def process_ravdess_dataset(
    dataset_dir="data/RAVDESS",
    output_dir="processed_features",
    min_segments=250,
    window_size=1.0,  # 1-second segments
    hop_size=0.5,     # 0.5-second hop
    max_videos=None,  # Maximum number of videos to process (None for unlimited)
    n_workers=None,   # Number of parallel workers (None for auto)
    emotions=None,    # List of emotion codes to process (None for all)
    actors=None,      # List of actor IDs to process (None for all)
    balance=False,    # Whether to balance examples across emotion categories
    visualize=False,  # Whether to visualize processed features
    validate=False,   # Whether to validate processed features
    process_all=False, # Whether to process all videos regardless of segment count
    skip_existing=False # Whether to skip already processed videos
):
    """Process RAVDESS dataset until at least min_segments are generated.
    
    Args:
        dataset_dir: Directory containing RAVDESS dataset
        output_dir: Directory to save processed features
        min_segments: Minimum number of segments to generate
        window_size: Time window size in seconds (1-second as requested)
        hop_size: Step size between windows (0.5 seconds as requested)
        max_videos: Maximum number of videos to process (None for all)
        n_workers: Number of parallel workers (None for auto-detect)
        emotions: List of emotion codes to process (None for all)
        actors: List of actor IDs to process (None for all)
        balance: Whether to balance examples across emotion categories
        visualize: Whether to visualize processed features
        validate: Whether to validate processed features
        
    Returns:
        Tuple of (segment_count, video_count) indicating the number of segments
        and videos processed.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualization directory if needed
    if visualize:
        os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    
    # Find all MP4 files in the dataset (RAVDESS uses MP4)
    video_pattern = os.path.join(dataset_dir, "**", "*.mp4")
    video_paths = glob.glob(video_pattern, recursive=True)
    
    if not video_paths:
        logging.error(f"No videos found matching pattern {video_pattern}")
        return 0, 0
    
    # Filter videos based on metadata (emotions, actors, balance)
    if emotions or actors or balance:
        original_count = len(video_paths)
        video_paths = filter_videos_by_metadata(video_paths, emotions, actors, balance)
        logging.info(f"Filtered from {original_count} to {len(video_paths)} videos based on metadata criteria")
        
        if not video_paths:
            logging.error("No videos left after filtering")
            return 0, 0
    
    # Shuffle videos to get a diverse selection of emotions and actors
    import random
    random.seed(42)  # For reproducibility
    random.shuffle(video_paths)
    
    # Limit the number of videos if specified
    if max_videos is not None:
        video_paths = video_paths[:max_videos]
    
    logging.info(f"Found {len(video_paths)} videos in {dataset_dir}")
    
    # Process videos sequentially until we reach min_segments
    segment_count = 0
    processed_videos = 0
    
    # Determine the number of workers
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)  # Leave one CPU free
    
    logging.info(f"Processing videos with {n_workers} workers")
    
    # Process videos and count segments
    # We'll use a small batch approach to monitor progress
    batch_size = min(100, len(video_paths))
    
    for batch_start in range(0, len(video_paths), batch_size):
        batch_end = min(batch_start + batch_size, len(video_paths))
        batch_videos = video_paths[batch_start:batch_end]
        
        logging.info(f"Processing batch of {len(batch_videos)} videos (videos {batch_start+1}-{batch_end} of {len(video_paths)})")
        
        if n_workers > 1:
            # Parallel processing with multiple workers
            with Pool(n_workers) as pool:
                args = [
                    (video_path, output_dir, "VGG-Face", window_size, hop_size, skip_existing)
                    for video_path in batch_videos
                ]
                
                output_files = list(tqdm(
                    pool.starmap(
                        process_video_wrapper,
                        args
                    ),
                    total=len(batch_videos),
                    desc="Processing videos"
                ))
        else:
            # Sequential processing
            output_files = []
            for video_path in tqdm(batch_videos, desc="Processing videos"):
                output_file = process_video_wrapper(
                    video_path, output_dir, "VGG-Face", window_size, hop_size, skip_existing
                )
                output_files.append(output_file)
        
        # Filter out None results and count successfully processed videos
        output_files = [f for f in output_files if f]
        processed_videos += len(output_files)
        
        # Count segments in output files
        batch_segment_count = count_segments_in_files(output_files)
        segment_count += batch_segment_count
        
        logging.info(f"Batch generated {batch_segment_count} segments, total now: {segment_count}")
        
        # Validate and visualize results if requested
        if validate or visualize:
            for npz_file in output_files:
                if validate:
                    is_valid = validate_processed_features(npz_file)
                    if not is_valid:
                        logging.warning(f"Validation failed for {npz_file}")
                
                if visualize:
                    vis_output_dir = os.path.join(output_dir, "visualizations")
                    vis_file = visualize_features(npz_file, vis_output_dir)
                    if vis_file:
                        logging.info(f"Created visualization: {vis_file}")
        
        # Check if we've reached the minimum number of segments (unless process_all is set)
        if not process_all and segment_count >= min_segments:
            logging.info(f"Reached target of {min_segments} segments after processing {processed_videos} videos")
            break
    
    logging.info(f"Final results: {segment_count} segments from {processed_videos} videos")
    
    return segment_count, processed_videos

def process_video_wrapper(video_path, output_dir, model_name, window_size, hop_size, skip_existing=False):
    """Wrapper around process_video_for_multimodal_lstm to handle exceptions.
    
    This allows us to continue processing even if one video fails.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save processed features
        model_name: DeepFace model to use
        window_size: Time window size in seconds
        hop_size: Step size between windows in seconds
        skip_existing: Whether to skip already processed videos
        
    Returns:
        Path to the saved feature file or None if processing fails or was skipped
    """
    # Check if output file already exists and skip_existing is True
    output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}.npz")
    if skip_existing and os.path.exists(output_file):
        logging.info(f"Skipping already processed video: {video_path}")
        return output_file
    
    try:
        return process_video_for_multimodal_lstm(
            video_path=video_path,
            output_dir=output_dir,
            model_name=model_name,
            window_size=window_size,
            hop_size=hop_size
        )
    except Exception as e:
        logging.error(f"Error processing {video_path}: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None

def count_segments_in_files(npz_files):
    """Count the total number of segments in a list of NPZ files."""
    segment_count = 0
    
    for npz_file in npz_files:
        try:
            data = np.load(npz_file, allow_pickle=True)
            
            if 'video_sequences' in data:
                # Count the number of sequences
                video_seqs = data['video_sequences']
                if isinstance(video_seqs, np.ndarray):
                    segment_count += len(video_seqs)
        except Exception as e:
            logging.error(f"Error counting segments in {npz_file}: {str(e)}")
    
    return segment_count

def filter_videos_by_metadata(video_paths, emotions=None, actors=None, balance=False):
    """Filter video paths based on metadata in filenames.
    
    Args:
        video_paths: List of video paths
        emotions: List of emotion codes to include (None for all)
        actors: List of actor IDs to include (None for all)
        balance: Whether to balance examples across emotion categories
        
    Returns:
        Filtered list of video paths
    """
    if not emotions and not actors and not balance:
        return video_paths
    
    filtered_paths = []
    emotion_counts = {}
    
    for path in video_paths:
        filename = os.path.basename(path)
        parts = filename.split('-')
        
        # Check if filename matches expected format
        if len(parts) < 3:
            logging.warning(f"Skipping file with unexpected format: {filename}")
            continue
        
        # Extract metadata
        try:
            actor_id = parts[0]
            emotion_code = parts[2]
            
            # Apply filters
            if emotions and emotion_code not in emotions:
                continue
                
            if actors and actor_id not in actors:
                continue
                
            # Track emotion counts for balancing
            if balance:
                if emotion_code not in emotion_counts:
                    emotion_counts[emotion_code] = []
                emotion_counts[emotion_code].append(path)
            else:
                filtered_paths.append(path)
                
        except Exception as e:
            logging.warning(f"Error parsing metadata from {filename}: {str(e)}")
            continue
    
    # If balancing, sample equal numbers from each emotion
    if balance and emotion_counts:
        # Find minimum count across all emotions
        min_count = min(len(paths) for paths in emotion_counts.values())
        
        # Force at least 5 examples per emotion if available
        min_count = max(min_count, min(5, min(len(paths) for paths in emotion_counts.values())))
        
        logging.info(f"Balancing dataset with {min_count} examples per emotion")
        
        # Sample from each emotion
        for emotion, paths in emotion_counts.items():
            import random
            random.seed(42)  # For reproducibility
            sampled_paths = random.sample(paths, min(min_count, len(paths)))
            filtered_paths.extend(sampled_paths)
            
            emotion_name = EMOTION_MAP.get(emotion, f"unknown-{emotion}")
            logging.info(f"Selected {len(sampled_paths)} examples for emotion: {emotion_name}")
    
    return filtered_paths

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process RAVDESS dataset for multimodal emotion recognition.')
    
    parser.add_argument('--dataset', type=str, default='data/RAVDESS',
                        help='Directory containing RAVDESS dataset')
    
    parser.add_argument('--output', type=str, default='processed_features_3_5s',
                        help='Directory to save processed features')
    
    parser.add_argument('--min-segments', type=int, default=1000,
                        help='Minimum number of segments to generate')
    
    parser.add_argument('--process-all', action='store_true',
                        help='Process all videos regardless of segment count')
    
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip videos that have already been processed')
    
    parser.add_argument('--window-size', type=float, default=3.5,
                        help='Time window size in seconds (default: 3.5 seconds)')
    
    parser.add_argument('--hop-size', type=float, default=0.5,
                        help='Step size between windows in seconds')
    
    parser.add_argument('--max-videos', type=int, default=None,
                        help='Maximum number of videos to process (None for all)')
    
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (None for auto-detect)')
    
    parser.add_argument('--emotions', type=str, default=None,
                        help='Comma-separated list of emotion codes to process (e.g., "03,04,05" for happy,sad,angry)')
    
    parser.add_argument('--actors', type=str, default=None,
                        help='Comma-separated list of actor IDs to process (e.g., "01,02,03")')
    
    parser.add_argument('--balance', action='store_true',
                        help='Balance examples across emotion categories')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize processed features')
    
    parser.add_argument('--validate', action='store_true',
                        help='Validate processed features')
    
    return parser.parse_args()

def main():
    """Main function to run the script."""
    # Fix multiprocessing for macOS
    try:
        set_start_method('spawn')
    except RuntimeError:
        # Method already set
        pass
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Process emotions and actors lists
    emotions = args.emotions.split(',') if args.emotions else None
    actors = args.actors.split(',') if args.actors else None
    
    # Number of parallel processes (leave one CPU free)
    n_workers = args.workers if args.workers is not None else max(1, cpu_count() - 1)
    
    print("\n" + "=" * 60)
    print(f"PROCESSING RAVDESS DATASET WITH OPENSMILE")
    print("=" * 60)
    print(f"Dataset directory: {args.dataset}")
    print(f"Output directory: {args.output}")
    print(f"Target segments: {args.min_segments}")
    print(f"Segment window size: {args.window_size} seconds (increased to 3.5s for better emotional context)")
    print(f"Segment hop size: {args.hop_size} seconds")
    print(f"Workers: {n_workers}")
    
    if args.process_all:
        print("Processing ALL videos regardless of segment count")
    if args.skip_existing:
        print("Skipping videos that have already been processed")
    
    if emotions:
        print(f"Processing only emotions: {[EMOTION_MAP.get(e, e) for e in emotions]}")
    if actors:
        print(f"Processing only actors: {actors}")
    if args.balance:
        print("Balancing examples across emotion categories")
    if args.visualize:
        print("Will visualize processed features")
    if args.validate:
        print("Will validate processed features")
        
    print("=" * 60 + "\n")
    
    # Process the dataset
    segment_count, video_count = process_ravdess_dataset(
        dataset_dir=args.dataset,
        output_dir=args.output,
        min_segments=args.min_segments,
        window_size=args.window_size,
        hop_size=args.hop_size,
        max_videos=args.max_videos,
        n_workers=n_workers,
        emotions=emotions,
        actors=actors,
        balance=args.balance,
        visualize=args.visualize,
        validate=args.validate,
        process_all=args.process_all,
        skip_existing=args.skip_existing
    )
    
    print("\n" + "=" * 60)
    print(f"PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Processed {video_count} videos")
    print(f"Generated {segment_count} segments")
    print(f"Data saved to {args.output}")
    
    if segment_count >= args.min_segments:
        print(f"\n✅ Successfully generated at least {args.min_segments} segments with OpenSMILE!")
    else:
        print(f"\n❌ Failed to generate enough segments. Review the logs for errors.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
