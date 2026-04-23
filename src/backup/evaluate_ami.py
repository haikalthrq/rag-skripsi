import os
import json
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
import nltk
from typing import List, Dict

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def load_json_chunks(filepath: str) -> List[str]:
    """Loads chunks from a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, list):
        # Case: ["chunk1", "chunk2", ...] (Ground Truth format)
        if all(isinstance(x, str) for x in data):
            return data
        # Case: [{"page_content": "...", ...}, ...] (LangChain/MaxMin format)
        elif all(isinstance(x, dict) and "page_content" in x for x in data):
            return [x["page_content"] for x in data]
        # Case: [{"text": "...", ...}, ...] (Alternative format)
        elif all(isinstance(x, dict) and "text" in x for x in data):
            return [x["text"] for x in data]
        # Case: MaxMin format with metadata sentences fallback
        elif all(isinstance(x, dict) for x in data):
            chunks = []
            for x in data:
                if "text" in x:
                    chunks.append(x["text"])
                elif "metadata" in x and "sentences" in x["metadata"]:
                    chunks.append(" ".join(x["metadata"]["sentences"]))
                else:
                    # Try to find any string field
                    found = False
                    for k, v in x.items():
                        if isinstance(v, str) and len(v) > 10:
                            chunks.append(v)
                            found = True
                            break
                    if not found:
                        chunks.append("") # Empty chunk
            return chunks
            
    raise ValueError(f"Unknown JSON format in {filepath}")

def get_sentence_labels(sentences: List[str], chunks: List[str]) -> List[int]:
    """
    Assigns each sentence to a chunk index.
    
    Args:
        sentences: List of atomic sentences.
        chunks: List of text chunks.
        
    Returns:
        List of chunk indices (labels) for each sentence.
    """
    labels = [-1] * len(sentences)
    
    # This is a simplified matching. 
    # Ideally, we should track character offsets.
    # Here we check if the sentence is a substring of the chunk.
    # Since chunks are composed of sentences, this should work for exact matches.
    
    # Optimization: Iterate through chunks and match sentences that belong to them.
    # Because chunks are ordered, we can maintain a cursor? 
    # No, let's just do a greedy search or exact match.
    
    # To handle potential whitespace differences, we strip.
    
    chunk_idx = 0
    
    for i, sent in enumerate(sentences):
        sent_clean = sent.strip()
        if not sent_clean:
            labels[i] = -2 # Ignore empty sentences
            continue
            
        found = False
        # Search in current chunk and next few chunks (to handle slight misalignments)
        # But usually, we iterate sequentially.
        
        # Let's try to find which chunk contains this sentence.
        # We search all chunks because sometimes order might be slightly weird 
        # (though it shouldn't be for text splitting).
        
        # Optimization: Start search from last found index
        for j in range(len(chunks)):
            # Check if sentence is in chunk
            # We use a loose check because of potential whitespace normalization differences
            if sent_clean in chunks[j]:
                labels[i] = j
                found = True
                break
        
        if not found:
            # Fallback: Try to match with less strictness (e.g. ignore spaces)
            for j in range(len(chunks)):
                if sent_clean.replace(" ", "") in chunks[j].replace(" ", ""):
                    labels[i] = j
                    found = True
                    break
                    
    return labels

def evaluate_ami(ground_truth_dir: str, prediction_dir: str):
    """
    Calculates AMI scores for all matching files in the directories.
    """
    gt_files = sorted([f for f in os.listdir(ground_truth_dir) if f.endswith("_gt.json")])
    
    results = []
    
    print(f"{'Filename':<60} | {'AMI Score':<10} | {'NMI Score':<10}")
    print("-" * 86)
    
    for gt_file in gt_files:
        # Construct corresponding prediction filename
        # GT: name_gt.json
        # Pred: name_chunks.json
        base_name = gt_file.replace("_gt.json", "")
        pred_file = f"{base_name}_chunks.json"
        
        gt_path = os.path.join(ground_truth_dir, gt_file)
        pred_path = os.path.join(prediction_dir, pred_file)
        
        if not os.path.exists(pred_path):
            print(f"Warning: Prediction file not found for {base_name}")
            continue
            
        try:
            # 1. Load Chunks
            gt_chunks = load_json_chunks(gt_path)
            pred_chunks = load_json_chunks(pred_path)
            
            print(f"  Debug: {base_name}")
            print(f"    GT Chunks: {len(gt_chunks)}")
            print(f"    Pred Chunks: {len(pred_chunks)}")

            # 2. Reconstruct "Full Text" to generate atomic sentences
            # We use the GT chunks to reconstruct the text because it represents the "ideal" document.
            full_text = " ".join(gt_chunks)
            
            # 3. Split into Atomic Units (Sentences)
            sentences = nltk.sent_tokenize(full_text)
            
            # 4. Assign Labels
            labels_true = get_sentence_labels(sentences, gt_chunks)
            labels_pred = get_sentence_labels(sentences, pred_chunks)
            
            # Filter out unmatched sentences (if any)
            valid_indices = [i for i in range(len(sentences)) if labels_true[i] != -1 and labels_pred[i] != -1]
            
            if not valid_indices:
                print(f"Error: No matching sentences found for {base_name}")
                continue
                
            y_true = [labels_true[i] for i in valid_indices]
            y_pred = [labels_pred[i] for i in valid_indices]
            
            # 5. Calculate AMI
            ami = adjusted_mutual_info_score(y_true, y_pred)
            nmi = normalized_mutual_info_score(y_true, y_pred)
            
            results.append({
                "filename": base_name,
                "ami": ami,
                "nmi": nmi
            })
            
            print(f"{base_name[:58]:<60} | {ami:.4f}     | {nmi:.4f}")
            
        except Exception as e:
            print(f"Error processing {base_name}: {e}")

    # Calculate Average
    if results:
        avg_ami = np.mean([r["ami"] for r in results])
        avg_nmi = np.mean([r["nmi"] for r in results])
        print("-" * 86)
        print(f"{'AVERAGE':<60} | {avg_ami:.4f}     | {avg_nmi:.4f}")

if __name__ == "__main__":
    # Paths
    GT_DIR = "data/ground_truth"
    PRED_DIR = "data/chunked/maxmin_semantic" # Change this to evaluate other methods
    
    print(f"Evaluating Ground Truth from: {GT_DIR}")
    print(f"Against Predictions from:     {PRED_DIR}")
    print("\n")
    
    evaluate_ami(GT_DIR, PRED_DIR)
