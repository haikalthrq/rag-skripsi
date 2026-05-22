import torch
import sys
from src.embedding.embedder import initialize_hf_embedder
from src.rag.generator import initialize_hf_generator
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("Loading embedder to GPU...")
embedder = initialize_hf_embedder(
    model_name="/workspace/rag-skripsi/models/Qwen3-Embedding-4B",
    device="cuda"
)
print(f"Embedder loaded. GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

print("Loading generator to GPU...")
generator = initialize_hf_generator(
    model_name="/workspace/rag-skripsi/models/Qwen3-4B-Instruct-2507",
    max_new_tokens=512
)
print(f"Generator loaded. GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
