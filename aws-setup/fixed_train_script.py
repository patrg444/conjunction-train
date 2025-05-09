#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test script to verify encoding is working correctly.
This will be used to test if we can execute Python with UTF-8 encoding.
"""

print("Encoding test successful - UTF-8 characters working: ©®µ§")
print("Proceeding to run the actual training...")

import subprocess
import os

# Try to fix the encoding in the original script
try:
    # Read the original script
    with open("scripts/train_branched_6class.py", "r") as f:
        content = f.read()

    # Add encoding declaration if not already present
    if "# -*- coding: utf-8 -*-" not in content:
        lines = content.split("\n")
        if lines[0].startswith("#!/usr/bin/env python"):
            # Insert after shebang line
            lines.insert(1, "# -*- coding: utf-8 -*-")
        else:
            # Insert at the beginning
            lines.insert(0, "# -*- coding: utf-8 -*-")
        
        fixed_content = "\n".join(lines)
        
        # Write back the fixed content
        with open("scripts/train_branched_6class_fixed.py", "w") as f:
            f.write(fixed_content)
        
        print("Created fixed script: scripts/train_branched_6class_fixed.py")
        
        # Run the fixed script
        subprocess.run(["python", "scripts/train_branched_6class_fixed.py"])
    else:
        print("Encoding declaration already exists, running original script")
        subprocess.run(["python", "scripts/train_branched_6class.py"])
        
except Exception as e:
    print(f"Error: {str(e)}")
    print("Falling back to original script execution")
    subprocess.run(["python", "scripts/train_branched_6class.py"])
