import os
import torch

def inspect_weights():
    checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "soil_classifier_model.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return

    print("=" * 65)
    print("      🔍 SOIL VISION MODEL CHECKPOINT INSPECTOR (PyTorch)")
    print("=" * 65)
    print(f"File Path    : {checkpoint_path}")
    print(f"File Size    : {os.path.getsize(checkpoint_path) / (1024*1024):.2f} MB\n")

    # Load state_dict (weights tensor map)
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    print(f"{'Layer Name':<42} | {'Tensor Shape':<18} | {'Parameters':<10}")
    print("-" * 75)

    total_params = 0
    for layer_name, tensor in state_dict.items():
        num_params = tensor.numel()
        total_params += num_params
        shape_str = str(list(tensor.shape))
        print(f"{layer_name:<42} | {shape_str:<18} | {num_params:<10,}")

    print("-" * 75)
    print(f"🎯 Total Trained Parameters (Weights & Biases): {total_params:,}")
    print(f"🎯 Total Layers with Trainable Tensors       : {len(state_dict)}")
    print("=" * 65)

    # Sample actual weight numbers from first convolutional filter
    first_conv_weight = state_dict['features.0.weight'][0, 0].cpu().numpy()
    print("\nSample Filter Weights (features.0.weight - First 3x3 Convolutional Kernel):")
    print(first_conv_weight)
    print("=" * 65)

if __name__ == "__main__":
    inspect_weights()
