import argparse
import pathlib
import csv

def get_stems(directory):
    """Return set of file stems (without .npy) in a directory."""
    return set(path.stem for path in pathlib.Path(directory).glob("*.npy"))

def main(text_dir, audio_dir, video_dir, output):
    text_stems = get_stems(text_dir)
    audio_stems = get_stems(audio_dir)
    video_stems = get_stems(video_dir)

    common_stems = text_stems & audio_stems & video_stems
    print(f"Found {len(common_stems)} entries present in all three modalities.")

    output_path = pathlib.Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text_path", "audio_path", "video_path"])
        for stem in sorted(common_stems):
            writer.writerow([
                stem,
                str(pathlib.Path(text_dir) / f"{stem}.npy"),
                str(pathlib.Path(audio_dir) / f"{stem}.npy"),
                str(pathlib.Path(video_dir) / f"{stem}.npy"),
            ])
    print(f"Wrote {len(common_stems)} rows to {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_dir", required=True)
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.text_dir, args.audio_dir, args.video_dir, args.output)
