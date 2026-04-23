import os
import json
import re
from pathlib import Path

def generate_ground_truth(
    input_dir="data/cleaned",
    output_dir="data/ground_truth",
    min_length=30
):
    """
    Generates 'Silver Standard' Ground Truth from cleaned text files.
    Assumes that the original document structure (paragraphs separated by double newlines)
    represents the 'ideal' chunking.
    """
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all .txt files
    files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    
    print(f"Found {len(files)} files to process.")
    
    for filename in files:
        input_path = os.path.join(input_dir, filename)
        output_filename = filename.replace(".txt", "_gt.json")
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"Processing {filename}...")
        
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            # Heuristic:
            # 1. Split by double newlines (assuming they mark paragraph boundaries)
            #    If the file was cleaned such that paragraphs are just single lines, 
            #    we might need to adjust. But usually \n\n is safe.
            
            # Check if file has \n\n
            if "\n\n" in text:
                raw_chunks = text.split("\n\n")
            else:
                # Fallback: If no double newlines, maybe every line is a paragraph?
                # Or maybe it's a raw dump. Let's assume single lines are paragraphs 
                # if they are long enough.
                print(f"  Warning: No double newlines found in {filename}. Splitting by single newline.")
                raw_chunks = text.split("\n")
            
            ground_truth_chunks = []
            
            for chunk in raw_chunks:
                # Clean up the chunk
                # Replace single newlines within a paragraph with space
                clean_chunk = chunk.replace("\n", " ").strip()
                
                # Remove multiple spaces
                clean_chunk = re.sub(r'\s+', ' ', clean_chunk)
                
                # Filter out noise (headers, page numbers, very short lines)
                if len(clean_chunk) >= min_length:
                    ground_truth_chunks.append(clean_chunk)
            
            # Save as JSON
            # Format: List of strings (chunks)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(ground_truth_chunks, f, indent=4, ensure_ascii=False)
                
            print(f"  Saved {len(ground_truth_chunks)} chunks to {output_filename}")
            
        except Exception as e:
            print(f"  Error processing {filename}: {e}")

if __name__ == "__main__":
    # Adjust paths relative to the script location or workspace root
    # Assuming script is run from workspace root
    generate_ground_truth()
