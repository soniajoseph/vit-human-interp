import json
import os
from pathlib import Path
from typing import Dict, List
import openai
from tqdm import tqdm

def load_results(results_path: str) -> Dict:
    """Load the JSON results file."""
    with open(results_path, 'r') as f:
        return json.load(f)

def get_gpt4_summary(captions: List[str]) -> str:
    """Get a summary of the captions using GPT-4."""
    prompt = f"""Please analyze these 20 image captions and provide a concise summary that identifies common themes, patterns, and shared characteristics across the images. Focus on recurring elements, similar objects, or consistent features that appear across multiple images.

Captions:
{chr(10).join(f"{i+1}. {caption}" for i, caption in enumerate(captions))}

Please provide a summary that:
1. Identifies common themes and patterns
2. Highlights shared characteristics across images

If there are no common themes or patterns or they are too far fetched, just say "No common themes or patterns found."
Reason about this in detail but then provide a concise summary sentence that is 1-3 sentences long. Format: "FINAL SUMMARY: <short summary sentences>"

Summary:"""

    # Updated API call for OpenAI client >= 1.0.0
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that analyzes image captions to identify patterns and themes."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )
    
    return response.choices[0].message.content.strip()

def save_results(results: Dict, output_path: str, mode='w'):
    """Save the enhanced results to a JSON file.
    
    Args:
        results: Dictionary of results to save
        output_path: Path to save the JSON file
        mode: 'w' for write (overwrite) or 'r+' for update
    """
    if mode == 'r+' and os.path.exists(output_path):
        # Read existing data
        with open(output_path, 'r') as f:
            existing_data = json.load(f)
        # Update with new data
        existing_data.update(results)
        results = existing_data
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

def process_results(results: Dict, output_path: str) -> Dict:
    """Process the results and add summaries for each neuron, saving after each request."""
    enhanced_results = {}
    
    for neuron_path, data in tqdm(results.items(), desc="Processing neurons"):
        # Create a copy of the existing data
        enhanced_data = data.copy()
        
        # Get all captions for this neuron
        captions = list(data['image_captions'].values())
        
        # Get summary from GPT-4
        summary = get_gpt4_summary(captions)
        
        # Add the summary to the enhanced data
        enhanced_data['summary'] = summary
        
        # Store in enhanced results
        enhanced_results[neuron_path] = enhanced_data
        
        # Save after each neuron is processed
        save_results({neuron_path: enhanced_data}, output_path, mode='r+')
    
    return enhanced_results

def main():
    # Set up OpenAI API key
    if 'OPENAI_API_KEY' not in os.environ:
        raise ValueError("Please set the OPENAI_API_KEY environment variable")
    
    # Define paths
    input_path = "vllms/vllm_results_single_images/vllm_results_single_images.json"
    output_path = "vllms/vllm_results_single_images/vllm_results_single_images_with_summaries.json"
    
    # Load results
    print("Loading results...")
    results = load_results(input_path)
    
    # Create empty output file
    save_results({}, output_path)
    
    # Process results and add summaries
    print("Processing results and generating summaries...")
    enhanced_results = process_results(results, output_path)
    print("Done!")

if __name__ == "__main__":
    main()