import os
import torch
from pathlib import Path
from tqdm import tqdm

def get_final_figure_filename(layer_idx, neuron_idx, layer_type, type_of_sampling, extreme_type):
    base_dir = f"layer_{layer_idx}/neuron_{neuron_idx}/{layer_type}/{type_of_sampling}/{extreme_type}"
    base_name = f"neuron_{neuron_idx}_layer_{layer_idx}_{extreme_type}_{type_of_sampling}_{layer_type}"
    return f"{base_dir}/{base_name}.png"

def count_png_files(base_path):
    """Count PNG files using the method from internvl2.5_one_file.py"""
    image_files = set()  # Changed to set for easier comparison
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('no_heatmap.png'):
                rel_path = str(Path(root) / file)
                image_files.add(rel_path)
    
    print("\nPNG File Analysis:")
    print(f"Total PNG files found: {len(image_files)}")
    
    # Analyze directory structure
    layer_counts = {}
    neuron_counts = {}
    for img_path in image_files:
        rel_parts = Path(img_path).relative_to(base_path).parts
        if len(rel_parts) >= 3:  # Assuming structure .../layer_X/neuron_Y/...
            layer = rel_parts[2]
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
            
            if len(rel_parts) >= 4:
                neuron = rel_parts[3]
                neuron_counts[neuron] = neuron_counts.get(neuron, 0) + 1
    
    print("\nBreakdown by layer:")
    for layer, count in sorted(layer_counts.items()):
        print(f"{layer}: {count} images")
    
    return image_files

def count_checkpoint_images(checkpoint_path, base_path):
    """Count images using the method from internvl2.5_caption_image.py and track image paths"""
    print("\nCheckpoint Analysis:")
    checkpoint = torch.load(checkpoint_path)
    total_images = 0
    strategy_counts = {}
    layer_type_counts = {}
    checkpoint_image_paths = set()  # Track all expected image paths from checkpoint
    
    for strategy in checkpoint.keys():
        strategy_counts[strategy] = 0
        for interval in checkpoint[strategy].keys():
            for layer_type in checkpoint[strategy][interval].keys():
                layer_type_counts[layer_type] = layer_type_counts.get(layer_type, 0)
                data = checkpoint[strategy][interval][layer_type]
                
                for layer_idx in range(len(data['activations'])):
                    num_neurons = min(15, data['activations'][layer_idx].shape[0])
                    for neuron_idx in range(num_neurons):
                        # Generate expected file path
                        expected_path = get_final_figure_filename(
                            layer_idx, neuron_idx, layer_type, strategy, interval)
                        full_path = str(Path(base_path) / expected_path)
                        checkpoint_image_paths.add(full_path)
                        
                        total_images += 1
                        strategy_counts[strategy] += 1
                        layer_type_counts[layer_type] += 1
    
    print(f"Total images in checkpoint: {total_images}")
    print("\nBreakdown by strategy:")
    for strategy, count in strategy_counts.items():
        print(f"{strategy}: {count} images")
    
    print("\nBreakdown by layer type:")
    for layer_type, count in layer_type_counts.items():
        print(f"{layer_type}: {count} images")
    
    return checkpoint_image_paths

def compare_file_lists(png_files, checkpoint_files):
    """Compare the two sets of files and print differences using only filenames"""
    print("\nFile Comparison Analysis:")
    
    # Convert full paths to just filenames
    png_filenames = {Path(f).name.replace('_no_heatmap', '') for f in png_files}
    checkpoint_filenames = {Path(f).name.replace('_no_heatmap', '') for f in checkpoint_files}
    
    # Files that exist on disk but aren't in checkpoint
    only_in_png = png_filenames - checkpoint_filenames
    if only_in_png:
        print(f"\nFilenames that exist on disk but aren't in checkpoint ({len(only_in_png)}):")
        for file in sorted(only_in_png)[:10]:  # Show first 10 examples
            print(f"  {file}")
        if len(only_in_png) > 10:
            print(f"  ... and {len(only_in_png) - 10} more")
    
    # Files that are in checkpoint but don't exist on disk
    only_in_checkpoint = checkpoint_filenames - png_filenames
    if only_in_checkpoint:
        print(f"\nFilenames expected from checkpoint but not found on disk ({len(only_in_checkpoint)}):")
        for file in sorted(only_in_checkpoint)[:10]:  # Show first 10 examples
            print(f"  {file}")
        if len(only_in_checkpoint) > 10:
            print(f"  ... and {len(only_in_checkpoint) - 10} more")
    
    # Files that match
    matching_files = png_filenames & checkpoint_filenames
    print(f"\nNumber of matching filenames: {len(matching_files)}")

def main():
    # Paths from the original scripts
    base_path = Path('/network/scratch/s/sonia.joseph/CLIP_AUDIT/sampled_images/open-clip_laion_CLIP-ViT-B-32-DataComp.XL-s13B-b90K/imagenet21k/train/all_neurons')
    checkpoint_path = '/home/mila/s/sonia.joseph/CLIP_AUDIT/clip_audit/activation_results/checkpoint_final.pt'
    
    print("Analyzing discrepancy in image counts...")
    
    # Count using both methods
    png_files = count_png_files(base_path)
    checkpoint_files = count_checkpoint_images(checkpoint_path, base_path)
    
    # Compare results
    print("\nComparison Results:")
    print(f"PNG files found: {len(png_files)}")
    print(f"Checkpoint images: {len(checkpoint_files)}")
    print(f"Difference: {abs(len(png_files) - len(checkpoint_files))} images")
    
    # Compare actual files
    compare_file_lists(png_files, checkpoint_files)

if __name__ == "__main__":
    main() 