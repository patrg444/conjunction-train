#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick fix for the Keras model saving issue in train_video_only_facenet_lstm_fixed.py
"""

import re
import os
import sys

def fix_keras_save_model_error(file_path):
    """Fix the Keras model saving issue in the training script."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Find the ModelCheckpoint callback and fix it
        pattern = r"ModelCheckpoint\(\s*os\.path\.join\(checkpoint_dir,\s*'best_model\.keras'\),\s*monitor='val_accuracy',\s*save_best_only=True,\s*mode='max',\s*verbose=1\s*\)"
        replacement = "ModelCheckpoint(\n            os.path.join(checkpoint_dir, 'best_model.h5'),\n            monitor='val_accuracy',\n            save_best_only=True,\n            mode='max',\n            verbose=1\n        )"
        
        # Apply the fix
        new_content = re.sub(pattern, replacement, content)
        
        # Also update any references to loading the best model
        new_content = new_content.replace(
            "best_model_path = os.path.join(checkpoint_dir, 'best_model.keras')",
            "best_model_path = os.path.join(checkpoint_dir, 'best_model.h5')"
        )
        
        # Write the fixed content back to the file
        with open(file_path, 'w') as f:
            f.write(new_content)
        
        print(f"Successfully fixed Keras model saving issue in {file_path}")
        return True
    except Exception as e:
        print(f"Error fixing Keras model saving issue: {e}")
        return False

if __name__ == "__main__":
    # Get the script path from command line or use default
    script_path = sys.argv[1] if len(sys.argv) > 1 else "scripts/train_video_only_facenet_lstm_fixed.py"
    fix_keras_save_model_error(script_path)
