import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm
import random

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# Initialize model and tokenizer
path = 'OpenGVLab/InternVL2_5-8B'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

# Setup paths
base_image_path = Path('/network/scratch/s/sonia.joseph/CLIP_AUDIT/selected_imagenet21k')
output_dir = Path('vllm_results_single_images')
output_dir.mkdir(exist_ok=True)
image_output_dir = output_dir / 'images'
image_output_dir.mkdir(exist_ok=True)

# Load the mapping file
with open('../figure_to_image_ids_mapping.json', 'r') as f:
    mapping = json.load(f)

# Base path for the PNG figures
base_figures_path = Path('/network/scratch/s/sonia.joseph/CLIP_AUDIT/sampled_images/open-clip_laion_CLIP-ViT-B-32-DataComp.XL-s13B-b90K/imagenet21k/train/all_neurons')

results = {}
generation_config = dict(max_new_tokens=1024, do_sample=False)
question = '<image>\nDescribe this image in detail. What are the main objects, their characteristics, and the overall scene or context?'

# Process each neuron visualization and its associated images
for viz_path, image_ids in tqdm(mapping.items(), desc="Processing neuron visualizations"):
    # Create directory structure based on viz_path
    viz_dir = image_output_dir / Path(viz_path.split('/')[-1])
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    results[viz_path] = {
        'neuron_info': {
            'layer': viz_path.split('/')[0],
            'neuron': viz_path.split('/')[2]
        },
        'image_captions': {}
    }

    # Copy the overall PNG figure
    png_path = base_figures_path / (viz_path[:-4] + '_no_heatmap.png')
    if png_path.exists():
        shutil.copy2(png_path, str(viz_dir) + '_no_heatmap.png')
    
    # Process each image ID for this neuron
    for image_id in tqdm(image_ids, desc=f"Processing images for {viz_path}", leave=False):
        try:
            # Construct full image path
            image_path = base_image_path / f"{image_id}.jpg"
            
            # Copy image to the corresponding directory
            output_image_path = viz_dir / f"{image_id}.jpg"
            if image_path.exists():
                shutil.copy2(image_path, output_image_path)
            
            # Load and process image
            pixel_values = load_image(str(image_path)).to(torch.bfloat16).cuda()
            
            # Get model response
            response = model.chat(tokenizer, pixel_values, question, generation_config)
            
            # Store results
            results[viz_path]['image_captions'][image_id] = response
            
            # Periodically save results
            with open(output_dir / 'vllm_results_single_images.json', 'w') as f:
                json.dump(results, f, indent=2)
                
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            results[viz_path]['image_captions'][image_id] = f"Error: {str(e)}"

# Final save of results
with open(output_dir / 'vllm_results_single_images.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"Processing complete. Results saved to {output_dir}")
