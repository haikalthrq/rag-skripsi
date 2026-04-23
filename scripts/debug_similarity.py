"""
Debug script to check actual similarity values from Qwen3 embeddings.
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

# Load GGUF model
from chunking.maxmin_chunker import initialize_embedding_model_gguf

def debug_similarity():
    # Load a sample file
    sample_file = "data/cleaned/statistik-pendidikan-2025.txt"
    
    print("Loading text...")
    with open(sample_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Tokenize to sentences
    sentences = sent_tokenize(text)
    print(f"Total sentences: {len(sentences)}")
    
    # Take first 20 sentences for debugging
    sample_sentences = sentences[:20]
    print(f"\nFirst 5 sentences:")
    for i, s in enumerate(sample_sentences[:5]):
        print(f"  {i}: {s[:80]}...")
    
    # Load model
    print("\nLoading Qwen3-Embedding model...")
    model = initialize_embedding_model_gguf(verbose=False)
    
    if model is None:
        print("ERROR: Failed to load model!")
        return
    
    # Get embeddings using model.embed()
    print("Generating embeddings...")
    embeddings_list = []
    for i, sentence in enumerate(sample_sentences):
        # Suppress stderr warnings
        stderr_fd = sys.stderr.fileno()
        with open(os.devnull, 'w') as devnull:
            old_stderr = os.dup(stderr_fd)
            os.dup2(devnull.fileno(), stderr_fd)
            try:
                emb = model.embed(sentence)
            finally:
                os.dup2(old_stderr, stderr_fd)
                os.close(old_stderr)
        embeddings_list.append(emb)
        if (i + 1) % 5 == 0:
            print(f"  Progress: {i + 1}/{len(sample_sentences)}")
    
    embeddings = np.array(embeddings_list)
    
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embeddings = embeddings / norms
    
    print(f"Embedding shape: {embeddings.shape}")
    
    # Compute pairwise similarity matrix
    print("\n=== SIMILARITY MATRIX (first 10x10) ===")
    sim_matrix = cosine_similarity(embeddings[:10], embeddings[:10])
    
    # Print as table
    print("     ", end="")
    for i in range(10):
        print(f"  S{i:02d}", end=" ")
    print()
    
    for i in range(10):
        print(f"S{i:02d}  ", end="")
        for j in range(10):
            if i == j:
                print(" --- ", end=" ")
            else:
                print(f"{sim_matrix[i,j]:.3f}", end=" ")
        print()
    
    # Statistics
    upper_triangle = sim_matrix[np.triu_indices(10, k=1)]
    print(f"\n=== STATISTICS ===")
    print(f"Min similarity:  {upper_triangle.min():.4f}")
    print(f"Max similarity:  {upper_triangle.max():.4f}")
    print(f"Mean similarity: {upper_triangle.mean():.4f}")
    print(f"Std similarity:  {upper_triangle.std():.4f}")
    
    # Percentiles
    print(f"\nPercentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  {p}th: {np.percentile(upper_triangle, p):.4f}")
    
    # Adjacent sentence similarities (what MaxMin uses)
    print(f"\n=== ADJACENT SIMILARITIES ===")
    for i in range(min(15, len(embeddings)-1)):
        sim = cosine_similarity(embeddings[i:i+1], embeddings[i+1:i+2])[0,0]
        print(f"  Sentence {i} <-> {i+1}: {sim:.4f}")
    
    print("\n=== CONCLUSION ===")
    print(f"With threshold 0.85:")
    print(f"  - Similarities above 0.85 will be MERGED")
    print(f"  - Similarities below 0.85 will be SPLIT")
    
    above_threshold = (upper_triangle > 0.85).sum()
    total = len(upper_triangle)
    print(f"  - {above_threshold}/{total} ({100*above_threshold/total:.1f}%) pairs above 0.85")

if __name__ == "__main__":
    debug_similarity()
