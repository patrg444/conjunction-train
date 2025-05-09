#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to fix the training script for Python 2.7 compatibility
"""
from __future__ import print_function  # For Python 2 compatibility
import sys
import os
import re

# Path to original script on the server
ORIGINAL_SCRIPT = "scripts/train_branched_6class.py"
# Path for fixed script
FIXED_SCRIPT = "scripts/train_branched_6class_fixed.py"

print("Python 2.7 compatibility fixer for training script")
print("=================================================")

def fix_script(input_file, output_file):
    print("Reading original script from: %s" % input_file)
    try:
        with open(input_file, 'r') as f:
            content = f.read()
        
        # Add encoding declaration if not present
        if "# -*- coding: utf-8 -*-" not in content:
            lines = content.split("\n")
            if lines[0].startswith("#!"):
                # Insert after shebang
                lines.insert(1, "# -*- coding: utf-8 -*-")
            else:
                # Insert at beginning
                lines.insert(0, "# -*- coding: utf-8 -*-")
            
            print("Added encoding declaration")
            content = "\n".join(lines)
        
        # Replace f-strings with old-style formatting
        # This is a simple version that handles basic cases
        f_string_pattern = r'f"([^"]*)"'
        content = re.sub(f_string_pattern, r'"%\1" % ', content)
        f_string_pattern2 = r"f'([^']*)'"
        content = re.sub(f_string_pattern2, r"'%\1' % ", content)
        
        # Fix common f-string patterns manually
        content = content.replace('f"', '"').replace('f\'', '\'')
        content = content.replace('{epoch}', '%(epoch)s').replace('{val_loss:.4f}', '%(val_loss).4f')
        
        # Check if script uses 'pathlib' which is Python 3 specific
        if "from pathlib import Path" in content:
            content = content.replace("from pathlib import Path", "import os")
            content = content.replace("Path(", "os.path.join(")
        
        print("Fixed Python 3 specific syntax")
        
        # Write the fixed content
        with open(output_file, 'w') as f:
            f.write(content)
        
        print("Fixed script written to: %s" % output_file)
        return True
    except Exception as e:
        print("Error fixing script: %s" % str(e))
        return False

# Fix the script
success = fix_script(ORIGINAL_SCRIPT, FIXED_SCRIPT)

if success:
    print("\nScript fixed successfully!")
    print("To run the fixed script, use:")
    print("  python %s" % FIXED_SCRIPT)
else:
    print("\nFailed to fix script. See errors above.")
    sys.exit(1)
