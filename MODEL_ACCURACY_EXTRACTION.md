# Model Validation Accuracy Extraction

This toolset allows you to extract validation accuracy metrics from all model training logs on the AWS training server. It's useful for comparing model performance, tracking progress, and identifying the best-performing models.

## Scripts Included

1. **extract_all_models_val_accuracy.py** - Python script to connect to AWS, extract and analyze model training logs.
2. **get_all_model_accuracies.sh** - Bash script wrapper for easy execution.
3. **compare_model_accuracies.py** - Python script to generate visual comparisons between models.

## Prerequisites

- SSH access to the AWS training server
- The AWS SSH key file (emotion-recognition-key-fixed-20250323090016.pem)
- Python 3 with required libraries (the script will attempt to install missing packages)

## Usage

### Using the Shell Script (Recommended)

Run the shell script to perform the extraction with interactive options:

```bash
./get_all_model_accuracies.sh
```

The script will:
1. Check prerequisites (SSH key, Python dependencies)
2. Test the connection to AWS
3. Ask if you want to generate accuracy plots
4. Run the extraction process
5. Report the results

### Using the Python Script Directly

You can also run the Python script directly with optional arguments:

```bash
./extract_all_models_val_accuracy.py [--plot]
```

Options:
- `--plot`: Generate accuracy plots for each model

## Output Files

The scripts generate several output files:

1. **model_validation_summary.csv** - CSV file containing a summary of validation metrics for all models, sorted by best validation accuracy
2. **model_validation_accuracy.json** - JSON file containing detailed metrics for all models
3. **model_accuracy_plots/** - Directory containing accuracy plots for each model (if --plot is specified)

## Example Output

The console output will display a summary table like:

```
SUMMARY OF MODEL VALIDATION ACCURACY
================================================================================
Model                                    Best Val Acc    Epoch     
--------------------------------------------------------------------------------
branched_regularization_sync_aug_tcn_large_fixed_v2   0.8477 (84.77%)  87       
branched_regularization_sync_aug_tcn_large_fixed      0.8338 (83.38%)  92       
branched_regularization_sync_aug_tcn_large            0.8182 (81.82%)  75       
branched_regularization_sync_aug                     0.7865 (78.65%)  63       
lstm_attention_sync_aug                              0.7744 (77.44%)  48       
lstm_attention                                       0.7689 (76.89%)  53       
================================================================================
```

## How It Works

The script:
1. Connects to the AWS instance via SSH
2. Finds all training log files (*.log) in the remote directory
3. Downloads each log file temporarily
4. Extracts validation accuracy metrics for each model
5. Computes summary statistics (best accuracy, best epoch, etc.)
6. Generates plots if requested
7. Summarizes results in console output and output files

## Model Comparison

After running the extraction, you can use the comparison tool to create visualizations that compare multiple models:

```bash
./compare_model_accuracies.py [--top N] [--min-epochs N] [--output filename.png]
```

Options:
- `--top N`: Show only the top N models by best validation accuracy (default: 5)
- `--min-epochs N`: Only include models with at least N epochs of training (default: 10)
- `--output`: Output filename for the comparison chart (default: model_comparison.png)

### Comparison Outputs

The comparison script generates several outputs:

1. **model_comparison.png** - Line chart showing validation accuracy over epochs for top models
2. **final_accuracies.png** - Bar chart comparing best validation accuracies
3. **model_performance_table.csv** - Detailed CSV with performance metrics for each model

### Example Usage

To compare the top 7 models with at least 30 epochs of training:

```bash
./compare_model_accuracies.py --top 7 --min-epochs 30
```

To compare all models with any number of epochs:

```bash
./compare_model_accuracies.py --top 100 --min-epochs 1
```

## Troubleshooting

If you encounter issues:

- Check SSH connectivity to the AWS instance (verify IP and key file)
- Verify Python dependencies (matplotlib, numpy, pandas)
- Ensure you have proper permissions for the key file (`chmod 400 key-file.pem`)
- Check for errors in the console output
- For comparison script, ensure model_validation_accuracy.json exists (run extraction first)
