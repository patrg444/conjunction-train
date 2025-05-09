import pandas as pd
import os

def combine_csv_files(train_file, val_file, test_file, output_file):
    """
    Combines train, validation, and test CSV files into a single CSV file.
    The header from the train_file is used.
    """
    try:
        df_train = pd.read_csv(train_file)
        df_val = pd.read_csv(val_file)
        df_test = pd.read_csv(test_file)

        # Concatenate dataframes. The header from the first df (train) will be used by default.
        combined_df = pd.concat([df_train, df_val, df_test], ignore_index=True)

        # Ensure the output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")

        combined_df.to_csv(output_file, index=False)
        print(f"Successfully combined files into: {output_file}")
        print(f"Total entries in combined file: {len(combined_df)}")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure all input files exist.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    train_manifest = "datasets/manifests/humor/temp_local_urfunny_raw_train.csv"
    val_manifest = "datasets/manifests/humor/temp_local_urfunny_raw_val.csv"
    test_manifest = "datasets/manifests/humor/temp_local_urfunny_raw_test.csv"
    combined_output_manifest = "datasets/manifests/humor/urfunny_raw_data_complete.csv"

    # Ensure the script is run from the project root for correct relative paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(project_root)
    print(f"Current working directory set to: {os.getcwd()}")

    combine_csv_files(train_manifest, val_manifest, test_manifest, combined_output_manifest)
