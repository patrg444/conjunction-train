import pandas as pd
import argparse

def filter_manifest_for_ravdess(input_manifest, output_manifest):
    """Filters a manifest CSV to keep only rows containing 'ravdess' in the path."""
    try:
        df = pd.read_csv(input_manifest)
        print(f"Read {len(df)} rows from {input_manifest}")

        # Keep only rows where the 'path' column contains 'ravdess'
        df_filtered = df[df['path'].str.contains('ravdess', case=False, na=False)]
        
        print(f"Filtered down to {len(df_filtered)} RAVDESS rows.")

        df_filtered.to_csv(output_manifest, index=False)
        print(f"Saved RAVDESS-only manifest to {output_manifest}")

    except FileNotFoundError:
        print(f"Error: Input manifest file not found at {input_manifest}")
    except KeyError:
        print("Error: Input manifest must contain a 'path' column.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter manifest CSV for RAVDESS entries.")
    parser.add_argument("input_manifest", help="Path to the input manifest CSV file.")
    parser.add_argument("output_manifest", help="Path to save the filtered output manifest CSV file.")
    args = parser.parse_args()

    filter_manifest_for_ravdess(args.input_manifest, args.output_manifest)
