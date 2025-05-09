#!/usr/bin/env python3
"""
Script to generate a robust humor dataset by:
1. Downloading multiple humor datasets from different sources
2. Creating proper train/validation splits based on source/talk_id
3. Ensuring no content-level duplication between splits
4. Generating datasets in the proper format for DistilBERT fine-tuning

Sources include:
- Short Jokes Dataset
- Funlines (from SemEval-2020 Task 7)
- UR-Funny multimodal humor recognition dataset
- SMILE dataset
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
import hashlib
import json
import requests
import zipfile
import gzip
import io
import re
import random
import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split


def download_with_progress(url, save_path):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    with open(save_path, 'wb') as f:
        with tqdm.tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {os.path.basename(save_path)}") as pbar:
            for data in response.iter_content(block_size):
                f.write(data)
                pbar.update(len(data))


def hash_text(text):
    """Create a hash of the text for exact duplicate detection"""
    if isinstance(text, str):
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    return "invalid"


def download_short_jokes():
