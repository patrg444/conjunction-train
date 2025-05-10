import pandas as pd
import os

splits_dir = "splits"
csv_files = ["train.csv", "val.csv", "test.csv"]
all_emotions = set()

print(f"Checking emotion labels in {splits_dir}...")

for filename in csv_files:
    filepath = os.path.join(splits_dir, filename)
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            if "emotion" in df.columns:
                # Map 'calm' to 'neutral' before finding unique values
                df["emotion"] = df["emotion"].replace("calm", "neutral")
                # Filter out 'surprise' if present
                df = df[df["emotion"] != "surprise"]

                unique_emotions = df["emotion"].unique()
                print(f"- {filename}: Found {len(unique_emotions)} unique emotions (after mapping/filtering): {sorted(list(unique_emotions))}")
                all_emotions.update(unique_emotions)
            else:
                print(f"- {filename}: 'emotion' column not found.")
        except Exception as e:
            print(f"- Error reading {filename}: {e}")
    else:
        print(f"- {filename}: File not found.")

print(f"\nOverall unique emotions found across all files (after mapping/filtering): {sorted(list(all_emotions))}")
print(f"Total unique emotions: {len(all_emotions)}")

# Compare with existing LABEL_MAP
LABEL_MAP_EXISTING = {
    "angry": 0, "disgust": 1, "fear": 2, "happy": 3, "neutral": 4, "sad": 5,
}
print(f"\nExisting LABEL_MAP keys: {sorted(list(LABEL_MAP_EXISTING.keys()))}")
print(f"Number of classes in existing LABEL_MAP: {len(LABEL_MAP_EXISTING)}")

missing_in_map = all_emotions - set(LABEL_MAP_EXISTING.keys())
extra_in_map = set(LABEL_MAP_EXISTING.keys()) - all_emotions

if not missing_in_map and not extra_in_map:
    print("\nConclusion: Existing LABEL_MAP matches the effective emotions found in CSV files.")
else:
    print("\nConclusion: LABEL_MAP needs review/update!")
    if missing_in_map:
        print(f"  - Emotions in CSVs (post-mapping) but MISSING from LABEL_MAP: {sorted(list(missing_in_map))}")
    if extra_in_map:
        print(f"  - Emotions in LABEL_MAP but NOT FOUND in CSVs (post-mapping): {sorted(list(extra_in_map))}")
