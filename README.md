# vit-human-interp
Analysis of Pareto data and VLM auto-interp, in preparation for NeurIPS 2025


## How to access the data


### Image data (top 20 arranged together)
* *Google Drive version.* The data that we gave Pareto for all the neurons is [here](https://drive.google.com/drive/u/0/folders/1THejTazygC8LhwGVshnB1CGid08kWpal). The random neurons that we gave Pareto are [here](https://drive.google.com/drive/u/0/folders/1hcainPL_2BmrP85cJ4RseTXrkNsfxb8Y).
* *Mila cluster version.* You can also access the data on the Mila cluster here: `/network/scratch/s/sonia.joseph/CLIP_AUDIT/sampled_images/open-clip_laion_CLIP-ViT-B-32-DataComp.XL-s13B-b90K/imagenet21k/train/all_neurons` and `/network/scratch/s/sonia.joseph/CLIP_AUDIT/sampled_images/open-clip_laion_CLIP-ViT-B-32-DataComp.XL-s13B-b90K/imagenet21k/train/all_neurons/random`. You should have access.

### Individual image data
* Load the image ids from `/network/scratch/s/sonia.joseph/CLIP_AUDIT/open-clip_laion_CLIP-ViT-B-32-DataComp.XL-s13B-b90K/imagenet21k/train/image_ids.h5`
* The relevant images corresponding to the image ids are under `/network/scratch/s/sonia.joseph/CLIP_AUDIT/selected_imagenet21k`
* For reference, [here is the code I used to generate the top 20 image figure from the individual images](https://github.com/soniajoseph/CLIP_AUDIT/blob/main/clip_audit/generate_imagenet21k_imgs_with_heatmaps.py).


### Human interp results
* **[Guide for Interpreters](https://docs.google.com/document/d/1w0rTnYypb3Nwi2VYwlHolECufMh68SaBbUZ_nHUfIow/edit?usp=drive_web&ouid=117036631744604853491)**
* **Mapping.** We randomized the file names so that the interpreters could do the evaluation blind. We have the mapping from the randomized names back to the original layer, sublayer, and neuron number [here](https://drive.google.com/file/d/1a1XvZACGqQtoc8gyEK3LNQFinDlDrqYP/view?usp=drive_link). The randomized file name for the random neurons is [here](https://drive.google.com/file/d/1b0N8ThQuqypMl_E1Sgchnf4EVPaksnsJ/view?usp=drive_link).
   
## Auto-Interp

### Setup

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. If running on the Mila cluster, execute these commands once you are on a compute node:
   ```bash
   module load python/3.10
   module load cuda/12.1.1
   ```

### Running InternVL2.5 Single-Image Captioning

The script `vllms/internvl2.5_caption_single_images.py` generates detailed captions for neuron visualization images using the InternVL2.5 model. 

#### Usage

```bash
python vllms/internvl2.5_caption_single_images.py
```

#### Arguments

- `--model_path`: Path to the InternVL2.5 model (default: 'OpenGVLab/InternVL2_5-8B')
- `--base_image_path`: Base path for input images
- `--base_figures_path`: Base path for PNG figures
- `--mapping_file`: Path to the mapping JSON file
- `--output_dir`: Output directory for results (default: 'vllm_results_single_images')
- `--input_size`: Input image size (default: 448)
- `--max_num`: Maximum number of image patches (default: 12)
- `--max_new_tokens`: Maximum number of new tokens for generation (default: 1024)

#### Output

The script creates the following outputs in the specified output directory:

1. A JSON file `vllm_results_single_images.json` containing:
   - Neuron information (layer and neuron number)
   - Generated captions for each image
   - Any errors encountered during processing

2. An `images` subdirectory containing:
   - Copied neuron visualization PNG files
   - Individual images associated with each neuron
