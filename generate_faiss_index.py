# Preprocess Audio file and Compute Embeddings
# Build retrieval database : Used for retrieving neighbors
# Build index for similarity search : Train and build a search index for querying neighbors.
import types
from tqdm import tqdm
import argparse
from omegaconf import OmegaConf
import faiss
import torch
import pandas as pd
from laion_clap import CLAP_Module

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
    # FAISS index
    index_types = config.index_args.index_types
    
    # function
    def make_index(embedding_dim, nlist):
        quantizer = faiss.IndexFlatL2(embedding_dim)
        index_cpu = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
        return index_cpu

    # nlist: 32~512, trade-off between search time, nprobe=32~128
    def create_indices(index_types, text_embedding_dim = 512, audio_embedding_dim = 768, nlist = 128):
        indices = {}
        if "pair" in index_types:
            pair_embedding_dim = text_embedding_dim + audio_embedding_dim

        for index_type in index_types:
            if index_type == "text":
                indices["text"] = make_index(text_embedding_dim, nlist)
            elif index_type == "audio":
                indices["audio"] = make_index(audio_embedding_dim, nlist)
            elif index_type == "pair":
                indices["pair"] = make_index(pair_embedding_dim, nlist)
            else:
                raise ValueError(f"Invalid index type: {index_type}")

        return indices
    
    selected_indices = create_indices(index_types)
    captions = []
    wav_paths = []
    
    # model
    clap = CLAP_Module(enable_fusion=True)  # 615M
    clap.load_ckpt()  # download the default pretrained checkpoint.
    
    def process_text_data(data, clap_model):
        return clap_model.get_text_embedding(x = data, use_tensor = True)
    def process_audio_data(data, clap_model):
        return clap_model.get_audio_embedding(x=data, use_tensor=True)

    modalities = {
        "text": {"process": process_text_data, "embeddings": [], "data_key": "caption"},
        "audio": {"process": process_audio_data, "embeddings": [], "data_key": "audio_sample"}
        # 새로운 모달을 추가할 수 있다.
    }
    
    # index types에 있는 종류를 embedding으로 바꾼다.
    with torch.no_grad():
        for batch in tqdm(dataloader): # batch -> audio_sample, caption, wav_path
            captions.extend(batch["caption"])
            wav_paths.extend(batch["wav_paths"])
            
            for modality, info in modalities.items():
                if modality in index_types:
                    data = batch[info["data_key"]]
                    outputs = info["process"](data, clap)
                    info["embeddings"].extend(outputs.cpu().contiguous())

    # faiss indices에 저장한다.
    for modality in index_types:
        embeddings = torch.cat(modalities[modality]["embeddings"]).numpy().astype('float32')
        selected_indices[modality].train(embeddings)
        selected_indices[modality].add(embeddings)
    
    # text_embeddings_list = []
    # def get_audio_embedding_patch(self, data):
    #     device = next(self.parameters()).device
    #     input_dict = {}
    #     keys = data[0].keys()
    #     for k in keys:
    #         input_dict[k] = torch.cat([d[k].unsqueeze(0) for d in data], dim=0).to(device)
    #     audio_embeds = self.encode_audio(input_dict, device=device)
    #     return audio_embeds["embedding"]

    # captions = []
    # wav_paths = []
    # audio_embeddings_list = []
    
    # clap.model.get_audio_embedding = types.MethodType(get_audio_embedding_patch, clap.model)
    # # Process each batch from the DataLoader
    # with torch.no_grad():
    #     for audio_sample, caption, wav_path in tqdm(dataloader):
    #         # inputs = processor(audios=audio_sample, return_tensors="pt").to(config.device)
    #         # outputs = model(**inputs)
    #         # pooler_output = outputs.pooler_output
    #         outputs = clap.get_audio_embedding_from_data(x=audio_sample, use_tensor=True) # B, 768
    #         audio_embeddings_list.extend(outputs.cpu().contiguous())
    #         captions.extend(caption)
    #         wav_paths.extend(wav_path)

    # audio_embeddings = torch.cat(audio_embeddings_list)
    # audio_embeddings = audio_embeddings.numpy().astype('float32')
    # index_cpu.train(audio_embeddings) # Suppose 1M, 5~10ms
    # index_cpu.add(audio_embeddings)
    
    return selected_indices, captions, wav_paths

def save_index(selected_indices, save_path, index_types, captions_list, wav_paths):
    for modality in index_types:
        if modality in selected_indices:
            index_save_path = f"{save_path}{modality}_faiss_index.bin"
            print(f"Faiss index for {modality} is ready with {selected_indices[modality].ntotal} vectors.")
            faiss.write_index(selected_indices[modality], index_save_path)
            print(f"Faiss index for {modality} saved to {index_save_path}")
            
    if captions_list and wav_paths:
        captions_df = pd.DataFrame({
            'caption': captions_list,
            'wav_path': wav_paths
        })
        captions_csv_path = "index2_caption_path.csv"
        captions_df.to_csv(captions_csv_path, index=False)
        print(f"Captions and wav paths saved to {captions_csv_path}")

    print("Saved all selected indices")

if __name__ == "__main__":
    config = get_config()
    # Load DataLoader in order : WavCaps(AudioCaps-SL,FreeSound, BBC SoundEffects, CLOTHO v2.1)
    # Process embeddings and concatenate into one tensor 48M samples can be processed in GPU memory
    dataloader = pretrain_dataloader(config, bucket=False, is_distributed=False, num_tasks=1, global_rank=0)
    selected_indices, captions_list, wav_paths = generate_faiss_index(config, dataloader)
    save_index(selected_indices, config.index_args.save_path, config.index_args.index_types, captions_list, wav_paths)



