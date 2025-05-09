#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_wav2vec2_emotion.py

Fine-tunes a pre-trained Wav2Vec2 model (e.g., facebook/wav2vec2-base)
on an emotion recognition dataset (RAVDESS + CREMA-D).

Uses the manifest file created by build_audio_manifest.py and the
Wav2VecDataset loader from dataloaders/wav2vec_dataset.py.

Includes support for:
- SpecAugment (applied after feature extraction)
- Class weighting for imbalanced datasets
- Freezing the feature extractor initially
- AdamW optimizer with cosine LR schedule + warmup
- Evaluation using Accuracy and Unweighted Average Recall (UAR)
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, recall_score, classification_report
from sklearn.model_selection import train_test_split

# HuggingFace imports
import datasets
import transformers
from datasets import load_dataset, load_metric
from transformers import (
    AutoConfig,
    AutoFeatureExtractor, # Replaces Wav2Vec2FeatureExtractor
    AutoModelForAudioClassification,
    Trainer,
    TrainingArguments,
    Wav2Vec2Processor, # Keep processor for tokenizer part if needed, else use FeatureExtractor
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

# Local imports
# Need to ensure dataloaders is in the Python path or use relative import
# Assuming script is run from project root:
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataloaders.wav2vec_dataset import Wav2VecDataset, collate_fn_pad

logger = logging.getLogger(__name__)

# --- Argument Parsing ---

@dataclass
class ModelArguments:
    """ Arguments pertaining to which model/config/tokenizer we are going to fine-tune from. """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"}
    )
    freeze_feature_extractor: bool = field(
        default=True, metadata={"help": "Whether to freeze the feature extractor layers of the model."}
    )
    freeze_encoder_layers: int = field(
        default=0, metadata={"help": "Number of encoder layers to freeze (0 means freeze none)."}
    )
    attention_dropout: float = field(
        default=0.1, metadata={"help": "The dropout ratio for the attention probabilities."}
    )
    activation_dropout: float = field(
        default=0.1, metadata={"help": "The dropout ratio for activations inside the fully connected layer."}
    )
    hidden_dropout: float = field(
        default=0.1, metadata={"help": "The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler."}
    )
    feat_proj_dropout: float = field(
        default=0.0, metadata={"help": "The dropout probabilitiy for the feature projection layer."} # Reduced from 0.1
    )
    mask_time_prob: float = field(
        default=0.075, # Increased from 0.05
        metadata={
            "help": "Probability of masking chunks of time steps in the input."
            "Applying SpecAugment AFTER feature extraction."
        },
    )
    mask_time_length: int = field(
        default=10,
        metadata={"help": "Length of mask for time steps."},
    )
    mask_feature_prob: float = field(
        default=0.1, # Increased from 0.0
        metadata={
            "help": "Probability of masking chunks of features (frequency bands) in the input."
            "Applying SpecAugment AFTER feature extraction."
        },
    )
    mask_feature_length: int = field(
        default=64, # Mask ~50% of 128 features
        metadata={"help": "Length of mask for features (frequency bands)."},
    )
    layerdrop: float = field(default=0.1, metadata={"help": "The LayerDrop probability."})
    ctc_loss_reduction: Optional[str] = field(
        default="mean", metadata={"help": "The way the ctc loss should be reduced. Should be one of 'mean' or 'sum'."}
    )


@dataclass
class DataTrainingArguments:
    """ Arguments pertaining to what data we are going to input our model for training and eval. """
    manifest_path: str = field(metadata={"help": "Path to the manifest TSV file (path<TAB>label_id)."})
    class_counts_json: Optional[str] = field(
        default=None, metadata={"help": "Path to JSON file with class counts for weighting."}
    )
    train_split_percentage: float = field(
        default=0.8, metadata={"help": "The percentage of the data to use for training."}
    )
    max_duration_in_seconds: float = field(
        default=10.0, metadata={"help": "Filter out audio longer than this duration."}
    )
    min_duration_in_seconds: float = field(
        default=0.5, metadata={"help": "Filter out audio shorter than this duration."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "The number of processes to use for the preprocessing."}
    )
    audio_column_name: str = field(
        default="path", metadata={"help": "The name of the dataset column containing the audio path."}
    )
    label_column_name: str = field(
        default="label_id", metadata={"help": "The name of the dataset column containing the labels."}
    )
    target_feature_size: int = field(default=1, metadata={"help": "The target feature size."}) # For processor


# --- Metrics Computation ---

def compute_metrics(pred):
    """Computes accuracy and Unweighted Average Recall (UAR)."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    acc = accuracy_score(labels, preds)
    uar = recall_score(labels, preds, average='macro') # Unweighted recall = UAR

    # Optional: Detailed report
    # report = classification_report(labels, preds, target_names=list(EMOTION_MAP.keys()), output_dict=True)
    # print(classification_report(labels, preds, target_names=list(EMOTION_MAP.keys())))

    return {
        'accuracy': acc,
        'uar': uar,
    }

# --- Custom Trainer for Class Weights ---

class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Compute custom loss using class weights
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device) if self.class_weights is not None else None)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# --- Main Function ---

def main():
    # --- Parse Arguments ---
    parser = transformers.HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # --- Setup Logging ---
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")

    # --- Set Seed ---
    set_seed(training_args.seed)

    # --- Load Manifest & Split ---
    logger.info(f"Loading manifest from {data_args.manifest_path}")
    all_entries = pd.read_csv(data_args.manifest_path, sep='\t', header=None, names=['path', 'label_id'])
    num_labels = all_entries['label_id'].nunique()
    logger.info(f"Found {len(all_entries)} total entries and {num_labels} unique labels.")

    # Stratified split
    train_entries, eval_entries = train_test_split(
        all_entries,
        test_size=1.0 - data_args.train_split_percentage,
        random_state=training_args.seed,
        stratify=all_entries['label_id']
    )
    train_entries = train_entries.reset_index(drop=True)
    eval_entries = eval_entries.reset_index(drop=True)

    # Save splits to temporary files for Wav2VecDataset
    train_manifest_path = Path(training_args.output_dir) / "train_manifest.tsv"
    eval_manifest_path = Path(training_args.output_dir) / "eval_manifest.tsv"
    train_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    train_entries.to_csv(train_manifest_path, sep='\t', header=False, index=False)
    eval_entries.to_csv(eval_manifest_path, sep='\t', header=False, index=False)
    logger.info(f"Split data: {len(train_entries)} train, {len(eval_entries)} eval.")
    logger.info(f"Saved temporary manifests to {training_args.output_dir}")

    # --- Load Processor/Feature Extractor ---
    # Use AutoFeatureExtractor for newer versions
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir, trust_remote_code=True
    )
    # Processor combines feature extractor and tokenizer (tokenizer not needed for classification)
    # processor = Wav2Vec2Processor.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)

    # --- Create Datasets ---
    logger.info("Creating train and eval datasets...")
    train_dataset = Wav2VecDataset(
        manifest_path=str(train_manifest_path),
        processor=feature_extractor, # Pass feature extractor
        max_duration_s=data_args.max_duration_in_seconds,
        min_duration_s=data_args.min_duration_in_seconds,
    )
    eval_dataset = Wav2VecDataset(
        manifest_path=str(eval_manifest_path),
        processor=feature_extractor, # Pass feature extractor
        max_duration_s=data_args.max_duration_in_seconds,
        min_duration_s=data_args.min_duration_in_seconds,
    )

    # --- Load Model Config ---
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task="wav2vec2_clf", # Standard task name
        cache_dir=model_args.cache_dir,
        trust_remote_code=True,
    )

    # --- Apply SpecAugment Config AFTER loading ---
    # These settings control the masking applied by the model itself during training
    setattr(config, 'mask_time_prob', model_args.mask_time_prob)
    setattr(config, 'mask_time_length', model_args.mask_time_length)
    setattr(config, 'mask_feature_prob', model_args.mask_feature_prob)
    setattr(config, 'mask_feature_length', model_args.mask_feature_length)
    setattr(config, 'attention_dropout', model_args.attention_dropout)
    setattr(config, 'activation_dropout', model_args.activation_dropout)
    setattr(config, 'hidden_dropout', model_args.hidden_dropout)
    setattr(config, 'feat_proj_dropout', model_args.feat_proj_dropout)
    setattr(config, 'layerdrop', model_args.layerdrop)

    # --- Load Model ---
    model = AutoModelForAudioClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        trust_remote_code=True,
    )

    # --- Freeze Layers ---
    if model_args.freeze_feature_extractor:
        model.freeze_feature_extractor() # Newer HF method
        # model.freeze_feature_encoder() # Older HF method
        logger.info("Froze Wav2Vec2 feature extractor.")

    if model_args.freeze_encoder_layers > 0:
         for i, layer in enumerate(model.wav2vec2.encoder.layers):
             if i < model_args.freeze_encoder_layers:
                 for param in layer.parameters():
                     param.requires_grad = False
         logger.info(f"Froze the first {model_args.freeze_encoder_layers} encoder layers.")


    # --- Class Weights ---
    class_weights = None
    if data_args.class_counts_json:
        try:
            with open(data_args.class_counts_json, 'r') as f:
                class_counts = json.load(f)
            # Ensure keys are integers if needed, and sort by key
            class_counts = {int(k): v for k, v in class_counts.items()}
            sorted_counts = [class_counts.get(i, 0) for i in range(num_labels)]

            # Calculate inverse frequency weights
            total_samples = sum(sorted_counts)
            weights = [total_samples / (num_labels * count) if count > 0 else 0 for count in sorted_counts]
            # Normalize weights
            weights_sum = sum(weights)
            class_weights = torch.tensor([w / weights_sum * num_labels for w in weights], dtype=torch.float) # Scale to keep loss magnitude similar
            logger.info(f"Loaded class counts: {dict(class_counts)}")
            logger.info(f"Computed class weights: {class_weights.tolist()}")
        except Exception as e:
            logger.error(f"Could not load or compute class weights from {data_args.class_counts_json}: {e}", exc_info=True)
            class_weights = None # Fallback to no weights
    else:
        logger.info("No class counts JSON provided, using unweighted loss.")


    # --- Initialize Trainer ---
    trainer_class = WeightedLossTrainer if class_weights is not None else Trainer
    trainer = trainer_class(
        model=model,
        data_collator=collate_fn_pad, # Use our custom collator
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=feature_extractor, # Pass feature extractor here for padding value access etc.
        class_weights=class_weights, # Pass weights to custom trainer
    )

    # --- Training ---
    logger.info("*** Training ***")
    # Detect last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    checkpoint = last_checkpoint if training_args.resume_from_checkpoint is None else training_args.resume_from_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()

    # Save processor/feature extractor and training arguments
    # Use feature_extractor here
    feature_extractor.save_pretrained(training_args.output_dir)
    # processor.save_pretrained(training_args.output_dir)
    # trainer.save_state() # Saves optimizer, scheduler, etc.

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # --- Evaluation ---
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(eval_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    # --- Clean up temporary manifests ---
    try:
        os.remove(train_manifest_path)
        os.remove(eval_manifest_path)
        logger.info("Removed temporary manifest files.")
    except OSError as e:
        logger.warning(f"Could not remove temporary manifest files: {e}")

    logger.info("Training finished.")


if __name__ == "__main__":
    main()
