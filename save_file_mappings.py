import os
import torch
import numpy as np
import h5py
from tqdm import tqdm

neuron_indices_mlp_out = np.load('/home/mila/s/sonia.joseph/CLIP_AUDIT/clip_audit/saved_data/clip_base_mlp_out.npy', allow_pickle=True).item()
neuron_indices_resid_post = np.load('/home/mila/s/sonia.joseph/CLIP_AUDIT/clip_audit/saved_data/clip_base_residual_post.npy', allow_pickle=True).item()
    
def get_final_figure_filename(layer_idx, neuron_idx, layer_type, type_of_sampling, extreme_type):
    # Get original neuron index from the loaded indices
    if layer_type == 'hook_mlp_out':
        orig_neuron_index = neuron_indices_mlp_out[layer_idx][neuron_idx]
    elif layer_type == 'hook_resid_post':
        orig_neuron_index = neuron_indices_resid_post[layer_idx][neuron_idx]
    else:
        orig_neuron_index = neuron_idx  # fallback if layer_type doesn't match

    base_name = f"neuron_{orig_neuron_index}_layer_{layer_idx}_{extreme_type}_{type_of_sampling}_{layer_type}"
    base_dir = f"layer_{layer_idx}/neuron_{neuron_idx}/{layer_type}/{type_of_sampling}/{extreme_type}"
    # os.makedirs(base_dir, exist_ok=True)
    return f"{base_dir}/{base_name}.png"

def extract_and_print_info(checkpoint_path, image_path):
    checkpoint = torch.load(checkpoint_path)
    total_images = 0
    for strategy in checkpoint.keys():
        for interval in checkpoint[strategy].keys():
            for layer_type in checkpoint[strategy][interval].keys():
                data = checkpoint[strategy][interval][layer_type]
                
                for layer_idx in range(len(data['activations'])):
                    num_neurons = min(15, data['activations'][layer_idx].shape[0])
                    
                    for neuron_idx in tqdm(range(num_neurons)):
                        image_ids = data['image_ids'][layer_idx][neuron_idx]
                        figure_filename = get_final_figure_filename(layer_idx, neuron_idx, layer_type, strategy, interval)
                        
                        print(f"Figure: {figure_filename}")
                        print(f"Image IDs: {image_ids}")
                        total_images += 1
    print(f"Total images: {total_images}")
    # Save mapping to JSON
    import json
    results = {}
    for strategy in checkpoint.keys():
        for interval in checkpoint[strategy].keys():
            for layer_type in checkpoint[strategy][interval].keys():
                data = checkpoint[strategy][interval][layer_type]
                
                for layer_idx in range(len(data['activations'])):
                    num_neurons = min(15, data['activations'][layer_idx].shape[0])
                    
                    for neuron_idx in range(num_neurons):
                        image_ids = data['image_ids'][layer_idx][neuron_idx]
                        figure_filename = get_final_figure_filename(layer_idx, neuron_idx, layer_type, strategy, interval)
                        results[figure_filename] = image_ids

    # Save to JSON file
    output_file = 'figure_to_image_ids_mapping.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nMapping saved to {output_file}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extract figure filenames and image IDs')
    parser.add_argument('--checkpoint', type=str, default='/home/mila/s/sonia.joseph/CLIP_AUDIT/clip_audit/activation_results/checkpoint_final.pt', help='Path to checkpoint file')
    parser.add_argument('--image-dir', type=str, default='/network/scratch/s/sonia.joseph/CLIP_AUDIT/selected_imagenet21k', help='Path to image directory')
    # parser.add_argument('--save-dir', type=str, default='/path/to/save/directory', help='Path to save figures')
    
    args = parser.parse_args()
    extract_and_print_info(args.checkpoint, args.image_dir)

if __name__ == "__main__":
    main()
