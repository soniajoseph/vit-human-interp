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
base_path = Path('/network/scratch/s/sonia.joseph/CLIP_AUDIT/sampled_images/open-clip_laion_CLIP-ViT-B-32-DataComp.XL-s13B-b90K/imagenet21k/train/all_neurons')
output_dir = Path('vllm_results_zero_shot_one_file')
output_dir.mkdir(exist_ok=True)
image_output_dir = output_dir / 'images'
image_output_dir.mkdir(exist_ok=True)

# Find all relevant PNG files
image_files = []
for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.endswith('no_heatmap.png'):
            image_files.append(Path(root) / file)

random.seed(42)
random.shuffle(image_files)
print(f"Total images: {len(image_files)}")

results = {}
generation_config = dict(max_new_tokens=1024, do_sample=False)
question = '<image>\nYou are shown 16 images that highly activated a neuron in a vision model. Describe any commonalities among all 16 images, what the neuron is encoding for. If the images have no commonalities at all, just say "No commonalities found". If there is groups of images that are similar, describe the commonalities of the groups.'

# Process each image
for img_path in tqdm(image_files[:20], desc="Processing images"):
    try:
        # Load and process image
        pixel_values = load_image(str(img_path), max_num=12).to(torch.bfloat16).cuda()
        
        # Get model response
        response = model.chat(tokenizer, pixel_values, question, generation_config)
        
        # Create relative path for storing in results
        rel_path = img_path.relative_to(base_path)
        
        # Store results
        results[str(rel_path)] = {
            'response': response,
            'neuron_info': {
                'layer': rel_path.parts[2],  # Assuming directory structure contains layer info
                'neuron': rel_path.stem.split('_')[1]  # Extract neuron number from filename
            }
        }
        
        # Copy image to output directory
        output_image_path = image_output_dir / rel_path.name
        shutil.copy2(img_path, output_image_path)
        
        # Periodically save results
        with open(output_dir / 'vllm_results_zero_shot_one_file.json', 'w') as f:
            json.dump(results, f, indent=2)
            
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")

# Final save of results
with open(output_dir / 'vllm_results_zero_shot_one_file.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"Processing complete. Results saved to {output_dir}")
