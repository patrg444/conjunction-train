# Audio-Only SOTA Emotion Recognition Plan (CREMA-D & RAVDESS)

This document outlines the execution plan to train and evaluate state-of-the-art (SOTA) audio-only speech emotion recognition (SER) models on the CREMA-D and RAVDESS datasets, targeting architectures like HuBERT-Large+Attention and Whisper Prompt-Tuning.

## PHASE 0 – PROJECT SETUP (Est: 30 mins)

*   **[0-1]** Create a new Git branch: `git checkout -b audio_sota_hubert`
*   **[0-2]** Install required Python libraries:
    ```bash
    pip install torch torchaudio torchmetrics pytorch_lightning torch_audiomentations transformers==4.39 einops rich
    pip install git+https://github.com/speechbrain/speechbrain.git  # For ECAPA-TDNN baseline
    ```
*   **[0-3]** Create the project directory structure:
    ```
    audio_sota/
      data/
      scripts/
      common/
      models/
      outputs/
      README.md
    ```

## PHASE 1 – DATA INGEST & PREPARATION (Est: 2 hours, overlaps with Phase 2)

*   **[1-1]** Verify raw dataset files (WAV format, 16kHz mono):
    *   CREMA-D: ~7442 clips
    *   RAVDESS: ~1440 clips
    *   Use `common/verify_raw.py` for checksums/counts.
*   **[1-2]** Build metadata CSVs (`metadata.csv`) for each dataset using `common/build_csv.py`. Include columns: `path`, `speaker`, `emotion`, `intensity`, `split`.
*   **[1-3]** Create speaker-independent train/validation/test splits (80/10/10) using `common/make_split.py` (leveraging `sklearn.model_selection.GroupShuffleSplit`). Save splits to `data/<dataset>/splits/`.
*   **[1-4]** Generate class distribution histograms using `common/class_hist.py` to inform potential use of focal loss or class weighting.

## PHASE 2 – FEATURE EXTRACTION (Est: 1 hour on GPU)

*   **[2-1]** Extract HuBERT-Large features using `scripts/extract_hubert.py`. Target model: `facebook/hubert-large-ls960-ft`. Save features as `.pt` files in `data/<dataset>/feats/hubert_large/`.
*   **[2-2]** (Optional) Extract 80-dimensional log-Mel spectrograms using `scripts/extract_mel80.py` for the ECAPA-TDNN baseline. Save as `.npy` in `data/<dataset>/feats/mel80/`.
*   **[2-3]** Configure a `dataset.yaml` or similar to point training scripts to the correct feature paths.

## PHASE 3 – MODEL & PIPELINE IMPLEMENTATION (Est: 0.5 day)

*   **[3-1]** Implement `common/datamodule.py`: PyTorch Lightning DataModule handling feature loading, batching, and on-the-fly augmentation via `torch_audiomentations`.
*   **[3-2]** Implement `common/augment.py`: Wrapper for the augmentation pipeline (SpecAugment, noise, gain, mixup) configured via YAML.
*   **[3-3]** Implement `models/hubert_attn.py`: HuBERT backbone + Bi-LSTM + Attentive Statistical Pooling head.
*   **[3-4]** Implement `models/whisper_prompt.py`: Whisper encoder + learnable prompt tokens + MLP classification head.
*   **[3-5]** Implement `models/ecapa_tdnn.py`: Wrapper around SpeechBrain's ECAPA-TDNN model.
*   **[3-6]** Create training script templates (`scripts/train_*.py`) using PyTorch Lightning, accepting CLI arguments for dataset, hyperparameters, checkpointing, etc. Include support for distributed training (`torchrun`).

## PHASE 4 – DRY RUN & DEBUGGING (Est: 1 hour)

*   **[4-1]** Run each training script (`train_hubert_attn.py`, `train_whisper_prompt.py`, `train_ecapa_tdnn.py`) for a small number of steps (e.g., 50) on a single GPU with `--dataset crema_d`.
*   **[4-2]** Verify forward/backward passes, loss calculation, and augmentation pipeline integration. Debug shape mismatches or dtype errors.

## PHASE 5 – FULL TRAINING RUNS (Est: Varies per model/GPU, ~4-8 hours per dataset)

*   **[5-A]** Launch full training for HuBERT-Attn on CREMA-D. Monitor via TensorBoard. Use early stopping based on validation UAR (Unweighted Average Recall).
*   **[5-B]** Launch full training for HuBERT-Attn on RAVDESS.
*   **[5-C]** Launch full training for Whisper Prompt-Tuning on both datasets.
*   **[5-D]** (Optional) Launch full training for ECAPA-TDNN baseline on both datasets.
*   Save best checkpoints based on validation UAR to `outputs/<runID>/checkpoints/`.

## PHASE 6 – EVALUATION & ANALYSIS (Est: 30 mins per model)

*   **[6-1]** Run `scripts/eval_model.py` on the test set using the best checkpoint for each model/dataset combination. Calculate WA, UAR, F1-macro, and generate confusion matrices. Save results to `outputs/<runID>/eval/`.
*   **[6-2]** Perform cross-corpus evaluation (e.g., test CREMA-D model on RAVDESS test set) to assess generalization.
*   **[6-3]** Compile key results into a summary table (`outputs/summary.csv`).

## PHASE 7 – (Optional) CROSS-VALIDATION & ENSEMBLING (Est: 0.5 day extra)

*   **[7-1]** Implement and run 5-fold cross-validation using `scripts/cv_train.py` for the best performing architecture (e.g., HuBERT-Attn).
*   **[7-2]** Evaluate an ensemble model (e.g., averaging logits from the 5-fold models) on the test set.

## PHASE 8 – EXPORT & DEPLOYMENT (Est: 1 hour)

*   **[8-1]** Export the best performing model(s) to TorchScript format using `scripts/export_ts.py`.
*   **[8-2]** (Optional) Package the model and inference logic into a pip-installable Python wheel.
*   **[8-3]** Upload final model artifacts (TorchScript model, label map, normalization stats) to a designated storage location (e.g., S3, Hugging Face Hub).

## Tracking Checklist

- [ ] PH0: Repo branch & env setup
- [ ] PH1: Metadata CSV & splits generated
- [ ] PH2: Feature extraction complete (HuBERT/WavLM)
- [ ] PH3: All model/common scripts implemented
- [ ] PH4: Dry-run passes successfully for all models
- [ ] PH5A: HuBERT-Attn CREMA training complete
- [ ] PH5B: HuBERT-Attn RAVDESS training complete
- [ ] PH5C: Whisper-Prompt training complete (both datasets)
- [ ] PH6: Evaluation metrics & confusion matrices generated
- [ ] PH6: Cross-corpus evaluation complete
- [ ] PH8: Best model(s) exported to TorchScript
