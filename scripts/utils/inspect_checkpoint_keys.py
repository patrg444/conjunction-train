import torch
import sys

def inspect_checkpoint_keys(checkpoint_path):
    """Loads a PyTorch checkpoint and prints the keys in its state_dict."""
    try:
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("\nKeys in 'state_dict':")
        else:
            state_dict = checkpoint
            print("\nKeys in checkpoint (no 'state_dict' key found):")

        for i, key in enumerate(state_dict.keys()):
            print(f"- {key}")
            if i >= 50: # Print only the first 50 keys to keep output manageable
                print("...")
                break

    except Exception as e:
        print(f"Error loading or inspecting checkpoint: {e}", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_checkpoint_keys.py <checkpoint_path>", file=sys.stderr)
        sys.exit(1)
    checkpoint_path = sys.argv[1]
    inspect_checkpoint_keys(checkpoint_path)
