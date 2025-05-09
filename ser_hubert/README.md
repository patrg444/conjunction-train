# HuBERT-based Speech Emotion Recognition (SER)

This project implements a Speech Emotion Recognition system using a pre-trained HuBERT model and PyTorch Lightning.

## Project Structure

```
ser_hubert/
├── data/                     # Placeholder for downloaded raw audio data (e.g., CREMA-D, RAVDESS)
├── splits/                   # Directory for train/val/test CSV manifests
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── hubert_ser_module.py      # PyTorch Lightning module for the SER model
├── data_module.py            # PyTorch Lightning DataModule for loading data
├── train.py                  # Main script for training the model
├── generate_embeddings.py    # Script to extract embeddings from a trained model
├── requirements.txt          # Python package dependencies
└── README.md                 # This file
```

## Setup

1.  **Create Conda Environment:**
    ```bash
    conda create -n ser_hubert python=3.11 -y
    conda activate ser_hubert
    ```

2.  **Install Dependencies:**
    Ensure you are in the `ser_hubert` conda environment. Use the specific python interpreter from the environment to install packages:
    ```bash
    # Replace /opt/anaconda3/envs/ser_hubert/bin/python with the actual path if different
    /opt/anaconda3/envs/ser_hubert/bin/python -m pip install -r requirements.txt
    ```
    *(Alternatively, if `requirements.txt` wasn't generated correctly or is missing, install manually):*
    ```bash
    /opt/anaconda3/envs/ser_hubert/bin/python -m pip install torch torchaudio transformers datasets evaluate accelerate audiomentations soundfile pytorch-lightning torchmetrics pandas
    ```

3.  **Prepare Data:**
    *   Download the CREMA-D and RAVDESS datasets (or your chosen datasets) and place the audio files in a location accessible by the scripts (e.g., inside `ser_hubert/data/`).
    *   Create CSV manifest files (`train.csv`, `val.csv`, `test.csv`) in the `ser_hubert/splits/` directory. Each CSV should have at least the following columns:
        *   `path`: Relative or absolute path to the audio file.
        *   `speaker`: Speaker ID.
        *   `emotion`: Emotion label string (e.g., "happy", "sad", "angry"). Ensure these labels match the keys in `LABEL_MAP` in `data_module.py`.
    *   Refer to Appendix B of the original document for an example script (`build_csv.py`) to generate these manifests. You might need to adapt it for your specific file structure and dataset combination.

## Training

Run the `train.py` script. Adjust hyperparameters as needed via command-line arguments.

```bash
# Activate the environment first
conda activate ser_hubert

# Run training (using the environment's python)
python train.py \
    --model_name "facebook/hubert-base-ls960" \
    --max_epochs 15 \
    --freeze_epochs 2 \
    --batch_size 8 \
    --accumulate_grad_batches 4 \
    --learning_rate 2e-5 \
    --precision "16-mixed" \
    --data_dir "splits" \
    --num_workers 4 \
    --pooling_mode "mean" \
    --dropout 0.1 \
    --early_stopping_patience 3
    # Add --accelerator gpu --devices 1 if using a GPU
```

Checkpoints will be saved in a `lightning_logs` directory based on the best validation UAR (Unweighted Average Recall).

## Generating Embeddings

After training, use `generate_embeddings.py` to extract pooled HuBERT embeddings from the last hidden layer for a specific data split using a trained checkpoint.

```bash
# Activate the environment first
conda activate ser_hubert

# Run embedding generation
python generate_embeddings.py \
    --checkpoint_path "path/to/your/best/checkpoint.ckpt" \
    --model_name "facebook/hubert-base-ls960" \
    --pooling_mode "mean" \
    --data_dir "splits" \
    --data_split "test" \
    --batch_size 16 \
    --output_file "test_embeddings.npz" \
    --accelerator "auto" # Or cpu, gpu, mps
```

This will save a `.npz` file containing the embeddings and corresponding labels.

## Notes

*   The `data_module.py` assumes the audio files referenced in the CSVs are accessible.
*   The `LABEL_MAP` in `data_module.py` should be updated to include all unique emotion labels present in your CSV files. The number of classes derived from this map must match the `num_classes` used when initializing `HubertSER`.
*   Ensure the `hubert_name` and `pooling_mode` arguments match between training and embedding generation.
*   The example training command uses mixed-precision (`16-mixed`). Adjust if necessary based on your hardware.
*   Consider adding a logger (e.g., `TensorBoardLogger`) to the `pl.Trainer` for better experiment tracking.
