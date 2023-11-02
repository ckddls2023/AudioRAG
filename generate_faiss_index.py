# Preprocess Audio file and Compute Embeddings
# Build retrieval database : Used for retrieving neighbors
# Build index for similarity search : Train and build a search index for querying neighbors.

import argparse
import torch
from omegaconf import OmegaConf
import faiss
from transformers import AutoProcessor, AutoFeatureExtractor, ClapModel, ClapAudioModel
import torch
import pandas as pd

from data_handling.pretrain_dataset import pretrain_dataloader

def get_config():
    parser = argparse.ArgumentParser(description="Generate Faiss index from PyTorch DataLoader")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    return config

def generate_faiss_index(config, dataloader):
    """
    Generate faiss index for a PyTorch DataLoader.

    Parameters:
    - dataloader: PyTorch DataLoader producing embeddings
    - embedding_dim: Dimension of the embedding

    Returns:
    - faiss index
    """
    # Initialize faiss index
    embedding_dim = 768
    index_cpu = faiss.IndexFlatL2(embedding_dim)
    res = faiss.StandardGpuResources()  # Set up GPU resources
    index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)
    model = ClapAudioModel.from_pretrained("laion/clap-htsat-fused").to(config.device)  # 615M
    processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")
    captions_list = []

    # Process each batch from the DataLoader
    for audio_sample, captions in dataloader:
        inputs = processor(audios=audio_sample, return_tensors="pt").to(config.device)
        outputs = model(**inputs)
        pooler_output = outputs.pooler_output
        index_gpu.add(pooler_output)
        captions_list.extend(captions)

    return index_gpu, captions_list

if __name__ == "__main__":
    config = get_config()

    # Load DataLoader in order : WavCaps(AudioCaps-SL,FreeSound, BBC SoundEffects, CLOTHO v2.1)
    # Process embeddings and concatenate into one tensor 48M samples can be processed in GPU memory
    dataloader = pretrain_dataloader(config, bucket=False, is_distributed=False, num_tasks=1, global_rank=0)
    faiss_index, captions_list = generate_faiss_index(config, dataloader)
    print("Faiss index is ready with", faiss_index.ntotal, "vectors.")
    # Save FAISS index DB
    save_path = "faiss_index.bin"
    faiss.write_index(faiss_index, save_path)
    print(f"Faiss index saved to {save_path}")
    # Save captions to a CSV
    captions_df = pd.DataFrame(captions_list, columns=["caption"])
    captions_csv_path = "captions.csv"
    captions_df.to_csv(captions_csv_path, index=False)


