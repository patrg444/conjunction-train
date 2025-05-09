import torch
import sys
import traceback
import torchvision

def run_test():
    print("--- Starting R3D-18 Forward Pass Test ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # --- R3D-18 Specific Setup ---
        print("Loading torchvision R3D-18 model...")
        # Load pretrained R3D-18 model
        weights = torchvision.models.video.R3D_18_Weights.DEFAULT
        model = torchvision.models.video.r3d_18(weights=weights)
        
        # Modify the final layer for binary classification (if needed, or just test feature extraction)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 2) # Example: 2 classes
        
        model = model.to(device)
        model.eval()
        print("R3D-18 model successfully loaded and moved to device.")
        
        # --- Input Tensor Creation ---
        print("Creating input tensor...")
        
        # R3D expects [B, C, T, H, W]
        batch_size = 1
        channels = 3
        total_frames = 16 # R3D typically uses shorter clips like 16 frames
        crop_size = 112 # R3D often uses 112x112
        
        # Create a single input tensor
        inputs = torch.randn(batch_size, channels, total_frames, crop_size, crop_size).to(device)
        
        # Display input shape for debugging
        print(f"Input tensor shape: {inputs.shape}")
        
        # --- Perform Forward Pass ---
        print("Performing forward pass...")
        with torch.no_grad():
            outputs = model(inputs)
            
        print(f"Forward pass successful! Output shape: {outputs.shape}")
        print("--- Test Completed Successfully ---")
        
    except ImportError as e:
         print(f"\nCRITICAL ERROR: Import failed - {e}", file=sys.stderr)
         print("Ensure torchvision is correctly installed.", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}", file=sys.stderr)
        print("\nDetailed traceback:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()
