import pandas as pd
import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

# Try to detect header
with open(args.input, "r") as f:
    first_line = f.readline().strip()
has_header = first_line.lower().startswith("path") or first_line.lower().startswith("datasets/")

# Read CSV
if has_header:
    df = pd.read_csv(args.input)
else:
    df = pd.read_csv(args.input, names=["path", "split", "label"])

# Duplicate path for audio/video, add id
df["audio_path"] = df["path"]
df["video_path"] = df["path"]
df["id"] = range(1, len(df) + 1)

# Reorder columns for extractor compatibility
out = df[["audio_path", "video_path", "split", "id", "label"]]
out.to_csv(args.output, index=False, header=True)
print(f"wrote {len(out)} rows to {args.output}")
