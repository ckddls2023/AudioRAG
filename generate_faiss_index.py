# Preprocess Audio file and Compute Embeddings
# Build retrieval database : Used for retrieving neighbors
# Build index for similarity search : Train and build a search index for querying neighbors.
from types import MethodType
from tqdm import tqdm
import argparse
import torch
from omegaconf import OmegaConf
import faiss
from transformers import AutoProcessor, ClapAudioModel
import torch
import laion_clap
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
    nlist = 128 # 32~512, trade-off between search time, nprobe=32~128
    quantizer = faiss.IndexFlatL2(embedding_dim)
    index_cpu = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
    # https://github.com/facebookresearch/faiss/wiki/Additive-quantizers
    # https://www.pinecone.io/learn/series/faiss/composite-indexes/
    # FlatL2 or IndexFlatIP where all the vectors are stored as without any compression or transformation.
    # LSH is an approximate nearest neighbor search where each vector is hashed to compact binary code
    # HNSW(Hierarchical Navigable Small World) neighbor search algorithm that builds a multi-layer graph structure
    # IVF is a method that partitions the vector space into several Voronoi cell, coarse-grained
    # PQ : Product Quantization, recall-(speed,memory) trade-off, fine-grained
    # IndexIVFFlat: vector space is partitioned into a Voronoi cell, nlist=number of centroid, nprobe=N close centroid
    # IndexIVFPQ: vector space is partitioned, m(embedding dim % m = 0, dim for Q), quantized(nbits)
    # When use complex faiss index with composition, please use index_factory. e.g) index_factory(d, "IVF256,PQ32x8")
    # model = ClapAudioModel.from_pretrained("laion/clap-htsat-fused").to(config.device)  # 615M
    # processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")
    captions = []
    wav_paths = []
    audio_embeddings_list = []
    def get_audio_embedding_patch(self, data):
        device = next(self.parameters()).device
        input_dict = {}
        keys = data[0].keys()
        for k in keys:
            input_dict[k] = torch.cat([d[k].unsqueeze(0) for d in data], dim=0).to(device)
        audio_embeds = self.encode_audio(input_dict, device=device)
        return audio_embeds["embedding"]

    clap = CLAP_Module(enable_fusion=True)  # 615M
    clap.load_ckpt()  # download the default pretrained checkpoint.
    clap.model.get_audio_embedding = types.MethodType(get_audio_embedding_patch, clap.model)
    # Process each batch from the DataLoader
    with torch.no_grad():
        for audio_sample, caption, wav_path in tqdm(dataloader):
            # inputs = processor(audios=audio_sample, return_tensors="pt").to(config.device)
            # outputs = model(**inputs)
            # pooler_output = outputs.pooler_output
            outputs = clap.get_audio_embedding_from_data(x=audio_sample, use_tensor=True) # B, 768
            audio_embeddings_list.extend(outputs.cpu().contiguous())
            captions.extend(caption)
            wav_paths.extend(wav_path)

    audio_embeddings = torch.cat(audio_embeddings_list)
    audio_embeddings = audio_embeddings.numpy().astype('float32')
    index_cpu.train(audio_embeddings) # Suppose 1M, 5~10ms
    index_cpu.add(audio_embeddings)
    return index_cpu, captions, wav_paths

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
    captions_df = pd.DataFrame({
        'caption': captions_list,
        'wav_path': wav_paths
    })
    captions_csv_path = "index2_caption_path.csv"
    captions_df.to_csv(captions_csv_path, index=False)


