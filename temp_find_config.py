import slowfast, os, inspect
try:
    pkg_dir = os.path.dirname(inspect.getfile(slowfast))
    print(f'SlowFast package dir: {pkg_dir}')
    count=0
    # Look for common Kinetics config files
    target_files=['slowfast_r50.yaml', 'slowfast_32x2_r50.yaml', 'slowfast_8x8_r50.yaml']
    found_paths = {}
    print('Searching for config files...')
    for root,_,files in os.walk(pkg_dir):
        for f in files:
            # Check lower case for flexibility
            if f.lower() in target_files:
                path = os.path.join(root, f)
                print(f'Found: {path}')
                # Store the first found path for each target file type
                if f.lower() not in found_paths:
                     found_paths[f.lower()] = path
                count += 1
    if count == 0:
        print('No standard SlowFast YAML config files found in package directory.')
    else:
        # Print the found paths clearly at the end
        print("\nFound config paths:")
        for name, path in found_paths.items():
            print(f"- {name}: {path}")

except ImportError:
    print("Error: Could not import the 'slowfast' package.")
except Exception as e:
    print(f"An error occurred: {e}")
