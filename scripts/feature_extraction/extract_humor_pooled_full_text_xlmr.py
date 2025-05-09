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

def load_text_model(model_name_or_path):
    logger.info(f"Loading XLM-RoBERTa model from: {model_name_or_path}")
    try:
        model = XLMRobertaModel.from_pretrained(model_name_or_path)
        tokenizer = XLMRobertaTokenizer.from_pretrained(model_name_or_path)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("XLM-RoBERTa model moved to CUDA.")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load XLM-RoBERTa model from {model_name_or_path}: {e}", exc_info=True)
        raise

def get_single_text_embedding(text, model, tokenizer, max_length=512):
    if not isinstance(text, str) or not text.strip():
        logger.warning("Empty or invalid text provided for embedding.")
        # Return a zero vector of the model's hidden size if text is invalid
        return np.zeros(model.config.hidden_size, dtype=np.float32)
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
        logger.error(f"Error extracting embedding for text '{text[:100]}...': {e}", exc_info=True)
        # Return a zero vector of the model's hidden size in case of error
        return np.zeros(model.config.hidden_size, dtype=np.float32)

def process_manifest_for_full_text_embeddings(
    manifest_df,
    text_model,
    text_tokenizer,
    embedding_dir,
    context_col='raw_context_text',
    punchline_col='raw_punchline_text',
    id_col='talk_id',
    overwrite=False
):
    os.makedirs(embedding_dir, exist_ok=True)
    logger.info(f"Saving pooled full text embeddings to: {embedding_dir}")
    
    xlmr_embedding_dim = text_model.config.hidden_size # Should be 1024 for XLM-R Large

    for idx, row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="Extracting Full Text Embeddings"):
        clip_id = str(row[id_col])
        output_filename = f"{clip_id}.npy"
        output_path = os.path.join(embedding_dir, output_filename)

        if os.path.exists(output_path) and not overwrite:
            continue

        context_text = str(row.get(context_col, "")).strip()
        punchline_text = str(row.get(punchline_col, "")).strip()

        # Concatenate context and punchline. Handle cases where one might be empty.
        if context_text and punchline_text:
            full_text = context_text + " " + punchline_text # Simple space concatenation
        elif context_text:
            full_text = context_text
        elif punchline_text:
            full_text = punchline_text
        else:
            logger.warning(f"Both context and punchline are empty for {clip_id}. Skipping embedding, will save zeros.")
            full_text = "" # This will lead to a zero vector from get_single_text_embedding

        embedding = get_single_text_embedding(full_text, text_model, text_tokenizer)
        
        if embedding is None or embedding.shape[0] != xlmr_embedding_dim:
            logger.warning(f"Failed to get valid embedding for {clip_id} (text: '{full_text[:50]}...'). Using zeros.")
            embedding = np.zeros(xlmr_embedding_dim, dtype=np.float32)
            
        np.save(output_path, embedding.astype(np.float32))

def main():
    parser = argparse.ArgumentParser(description='Extract pooled text embeddings for concatenated context and punchline using XLM-RoBERTa.')
    parser.add_argument('--manifest_path', type=str, required=True, 
                        help='Path to the input manifest CSV (e.g., datasets/manifests/humor/urfunny_raw_data_complete.csv).')
    parser.add_argument('--text_model_name_or_path', type=str, default='xlm-roberta-large', 
                        help='Hugging Face model name or path for XLM-RoBERTa (e.g., "xlm-roberta-large").')
    parser.add_argument('--embedding_dir', type=str, required=True, 
                        help='Directory to save extracted text embeddings (e.g., datasets/humor_embeddings_v2/text_pooled_full_xlmr/).')
    parser.add_argument('--context_col', type=str, default='raw_context_text', 
                        help='Column name in manifest for the context text.')
    parser.add_argument('--punchline_col', type=str, default='raw_punchline_text', 
                        help='Column name in manifest for the punchline text.')
    parser.add_argument('--id_col', type=str, default='talk_id', 
                        help='Column name for unique IDs.')
    parser.add_argument('--overwrite', action='store_true', 
                        help='Overwrite existing embeddings.')
    
    args = parser.parse_args()

    try:
        manifest_df = pd.read_csv(args.manifest_path)
        logger.info(f"Loaded manifest: {args.manifest_path} with {len(manifest_df)} entries.")
    except Exception as e:
        logger.error(f"Failed to load manifest file {args.manifest_path}: {e}", exc_info=True)
        sys.exit(1)

    text_model, text_tokenizer = load_text_model(args.text_model_name_or_path)
    
    xlmr_hidden_size = text_model.config.hidden_size
    logger.info(f"XLM-R hidden size: {xlmr_hidden_size}. This will be the dimension of the output embeddings.")

    process_manifest_for_full_text_embeddings(
        manifest_df,
        text_model,
        text_tokenizer,
        args.embedding_dir,
        context_col=args.context_col,
        punchline_col=args.punchline_col,
        id_col=args.id_col,
        overwrite=args.overwrite
    )
    logger.info("Full text embedding extraction complete.")

if __name__ == '__main__':
    main()
