#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Very simple test script that should work with any Python version.
"""

from __future__ import print_function  # For Python 2 compatibility

print("UTF-8 encoding test successful: ©®µ§")
print("Python environment check...")

import sys
import os
import platform

print("Python version: %s" % sys.version)
print("Python executable: %s" % sys.executable)
print("OS Platform: %s" % platform.platform())

# Test basic file operations
with open('version_test.txt', 'w') as f:
    f.write("Test file written with Python %s" % sys.version)

print("Test file written successfully.")

# Check what's in the current directory
print("\nCurrent directory contents:")
for item in os.listdir('.'):
    print("- %s" % item)

print("\nTest complete. If you see this, basic Python is working with proper encoding.")
print("Next step: Apply encoding fix to the main script:")
print("1. Add '# -*- coding: utf-8 -*-' as the second line of the script.")
print("2. Use older string formatting (%s) instead of f-strings.")
