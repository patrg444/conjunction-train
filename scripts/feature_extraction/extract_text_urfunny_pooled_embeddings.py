#!/usr/bin/env python
import os
import argparse
import pandas as pd
import numpy as np
import torch
import logging
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from tqdm import tqdm
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_text_model(model_path):
    logger.info(f"Loading XLM-RoBERTa model from: {model_path}")
    try:
        model = XLMRobertaModel.from_pretrained(model_path)
        tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("XLM-RoBERTa model moved to CUDA.")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load XLM-RoBERTa model from {model_path}: {e}", exc_info=True)
        raise

def get_text_embedding(text, model, tokenizer, max_length=512):
    if not isinstance(text, str) or not text.strip():
        logger.warning("Empty or invalid text provided for embedding.")
        return None
    try:
        inputs = tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        )
        if torch.cuda.is_available():
            inputs = {key: val.cuda() for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pool the last hidden state
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embedding.squeeze() # Squeeze to remove batch dim if batch_size is 1
    except Exception as e:
        logger.error(f"Error extracting embedding for text '{text[:50]}...': {e}", exc_info=True)
        return None

def process_manifest_text(
    manifest_df,
    text_model,
    text_tokenizer,
    embedding_dir,
    text_col='transcript',
    id_col='talk_id',
    delimiter=' ||| ',
    overwrite=False,
    target_dim=2048 # Expected: 1024 (context) + 1024 (punchline)
):
    os.makedirs(embedding_dir, exist_ok=True)
    logger.info(f"Saving pooled text embeddings to: {embedding_dir}")
    
    xlmr_embedding_dim = text_model.config.hidden_size # Should be 1024 for XLM-R Large

    for idx, row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="Extracting Text Embeddings"):
        clip_id = str(row[id_col])
        output_filename = f"{clip_id}.npy"
        output_path = os.path.join(embedding_dir, output_filename)

        if os.path.exists(output_path) and not overwrite:
            continue

        if text_col not in row or not isinstance(row[text_col], str) or not row[text_col].strip():
            logger.warning(f"Missing or invalid transcript for {clip_id} in column '{text_col}'. Skipping.")
            # Save a zero vector of the target dimension if text is missing
            np.save(output_path, np.zeros(target_dim, dtype=np.float32))
            continue

        full_transcript = row[text_col]
        context_text = ""
        punchline_text = ""

        if delimiter in full_transcript:
            parts = full_transcript.split(delimiter, 1)
            context_text = parts[0].strip()
            punchline_text = parts[1].strip()
        else:
            logger.warning(f"Delimiter '{delimiter}' not found in transcript for {clip_id}. Using full transcript as context.")
            context_text = full_transcript.strip()
            # punchline_text will remain empty

        context_embedding = get_text_embedding(context_text, text_model, text_tokenizer)
        if context_embedding is None or context_embedding.shape[0] != xlmr_embedding_dim:
            logger.warning(f"Failed to get valid context embedding for {clip_id}. Using zeros.")
            context_embedding = np.zeros(xlmr_embedding_dim, dtype=np.float32)

        punchline_embedding = np.zeros(xlmr_embedding_dim, dtype=np.float32) # Default to zeros
        if punchline_text: # Only process if punchline exists
            punchline_embedding_extracted = get_text_embedding(punchline_text, text_model, text_tokenizer)
            if punchline_embedding_extracted is not None and punchline_embedding_extracted.shape[0] == xlmr_embedding_dim:
                punchline_embedding = punchline_embedding_extracted
            else:
                logger.warning(f"Failed to get valid punchline embedding for {clip_id}. Using zeros for punchline.")
        
        combined_embedding = np.concatenate((context_embedding, punchline_embedding))

        if combined_embedding.shape[0] == target_dim:
            np.save(output_path, combined_embedding.astype(np.float32))
        else:
            logger.error(f"Final embedding for {clip_id} has incorrect shape {combined_embedding.shape}, expected {target_dim}. Saving zeros.")
            np.save(output_path, np.zeros(target_dim, dtype=np.float32))


def main():
    parser = argparse.ArgumentParser(description='Extract context and punchline pooled text embeddings for UR-FUNNY.')
    parser.add_argument('--manifest_path', type=str, required=True, help='Path to the input manifest CSV (e.g., ur_funny_train_humor_cleaned.csv).')
    parser.add_argument('--text_model_path', type=str, required=True, help='Path to Hugging Face XLM-RoBERTa model directory.')
    parser.add_argument('--embedding_dir', type=str, required=True, help='Directory to save extracted text embeddings.')
    parser.add_argument('--text_col', type=str, default='transcript', help='Column name in manifest for the full transcript.')
    parser.add_argument('--id_col', type=str, default='talk_id', help='Column name for unique IDs.')
    parser.add_argument('--delimiter', type=str, default=' ||| ', help='Delimiter for context and punchline.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing embeddings.')
    
    args = parser.parse_args()

    try:
        manifest_df = pd.read_csv(args.manifest_path)
        logger.info(f"Loaded manifest: {args.manifest_path} with {len(manifest_df)} entries.")
    except Exception as e:
        logger.error(f"Failed to load manifest file {args.manifest_path}: {e}", exc_info=True)
        sys.exit(1)

    text_model, text_tokenizer = load_text_model(args.text_model_path)
    
    # Determine target dimension based on XLM-R model's hidden size (e.g., 1024 * 2 for concatenated context/punchline)
    xlmr_hidden_size = text_model.config.hidden_size
    target_embedding_dim = xlmr_hidden_size * 2 
    logger.info(f"XLM-R hidden size: {xlmr_hidden_size}. Target concatenated embedding dimension: {target_embedding_dim}")

    process_manifest_text(
        manifest_df,
        text_model,
        text_tokenizer,
        args.embedding_dir,
        text_col=args.text_col,
        id_col=args.id_col,
        delimiter=args.delimiter,
        overwrite=args.overwrite,
        target_dim=target_embedding_dim
    )
    logger.info("Text embedding extraction complete.")

if __name__ == '__main__':
    main()
