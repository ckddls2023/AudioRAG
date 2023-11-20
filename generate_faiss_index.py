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
import time
import os
import librosa
from laion_clap import CLAP_Module

from data_handling.retrieval_dataset import RetrievalIndex
from data_handling.pretrain_dataset import pretrain_dataloader

def check_and_create_directory(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Directory created: {save_path}")
    else:
        print(f"Directory already exists: {save_path}")

def get_config():
    parser = argparse.ArgumentParser(description="Generate Faiss index from PyTorch DataLoader")
    parser.add_argument("--config", type=str, default="./configs/pretrain.yaml", help="Path to the config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    return config

def process_text_data(data, clap_model, config):
    return clap_model.get_text_embedding(x=data, use_tensor=True)

def process_audio_data(data, clap_model, config):
    if config.index_args.audio_dimension == 768: # before projection
        def get_audio_embedding_before_projection(self, data):
                """Get the audio embedding from the model

                Parameters
                ----------
                data: a list of dict
                    the audio input dict list from 'get_audio_feature' method

                Returns
                ----------
                audio_embed: torch.Tensor
                    a tensor of audio_embeds (N, D)

                """
                device = next(self.parameters()).device
                input_dict = {}
                keys = data[0].keys()
                for k in keys:
                    input_dict[k] = torch.cat([d[k].unsqueeze(0) for d in data], dim=0).to(device)
                audio_embeds = self.encode_audio(input_dict, device=device)["embedding"]
                return audio_embeds
        clap_model.model.get_audio_embedding = types.MethodType(get_audio_embedding_before_projection, clap_model.model)
    return clap_model.get_audio_embedding_from_data(x=data, use_tensor=True)


def generate_faiss_index(config, dataloader):
    """
    Generate faiss index for a PyTorch DataLoader.

    Parameters:
    - dataloader: PyTorch DataLoader producing embeddings
    - embedding_dim: 512차원으로 통일했습니다. / audio만 768차원이 사용 가능하다. 
    - pretrain.yaml 파일
    index_args:
    index_save_path: "./data/index"
    index_types: ["audio"]
    audio_dimension: 768 # 768 or 512
    text_dimension: 512 # 512

    Returns:
    - captions N개, wav_paths N개, selected_indices는 index_types에서 선택한 index를 생성할 수 있습니다.
    - AudioRAG 폴더에 caption_wav_path.csv(captions, wav_paths)가 저장되고, AudioRAG/data/index 폴더에 audio_faiss_index.bin, text_faiss_index.bin이 저장됩니다.
    """

    # FAISS index
    index_types = config.index_args.index_types

    # function
    def make_index(embedding_dim, nlist):
        quantizer = faiss.IndexFlatL2(embedding_dim)
        index_cpu = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
        return index_cpu

    # nlist: 32~512, trade-off between search time, nprobe=32~128
    def create_indices(index_types, text_embedding_dim=config.index_args.text_dimension, audio_embedding_dim=config.index_args.audio_dimension, nlist=128):
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
    clap = CLAP_Module(enable_fusion=True, device = 'cuda:3')  # 615M
    clap.load_ckpt()  # download the default pretrained checkpoint.

    modalities = {
        "text": {"process": process_text_data, "embeddings": [], "data_key": "caption"},
        "audio": {"process": process_audio_data, "embeddings": [], "data_key": "audio_sample"}
        # 새로운 모달을 추가할 수 있다.
    }

    # index types에 있는 종류를 embedding으로 바꾼다.
    # for test with samples
    # from itertools import islice
    # num_batches_to_test = 20
    # tqdm(islice(dataloader, num_batches_to_test)):
    
    name_map = {
        'audio_sample': 0,
        'caption': 1,
        'wav_path': 2
    }
    
    with torch.no_grad():
        start_time = time.time()
        for batch in tqdm(dataloader):  # batch -> audio_sample, caption, wav_path
            captions.extend(batch[name_map["caption"]])
            wav_paths.append(batch[name_map["wav_path"]])

            for modality, info in modalities.items():
                if modality in index_types:
                    data = batch[name_map[info["data_key"]]]
                    outputs = info["process"](data, clap, config)
                    info["embeddings"].extend(outputs.cpu().contiguous())

    # faiss indices에 저장한다.
    for modality in index_types:
        embeddings = torch.stack(modalities[modality]["embeddings"]).numpy().astype('float32')
        # (option) if normalized, faiss.normalize()
        selected_indices[modality].train(embeddings)
        selected_indices[modality].add(embeddings)
        elapsed_time = time.time() - start_time
        print(f"elapsed time for {modality}_faiss_index: {elapsed_time}")

    wav_paths_list = [item for sublist in wav_paths for item in sublist]
    return selected_indices, captions, wav_paths_list 

def save_index(selected_indices, captions_list, wav_paths, config, mode = 'pretrain'):
    save_path = config.index_args.index_save_path
    index_types = config.index_args.index_types
    dimension = {
        "audio": config.index_args.audio_dimension,
        "text": config.index_args.text_dimension
    }
    for modality in index_types:
        if modality in selected_indices:
            index_save_path = f"{save_path}/{modality}_{dimension[modality]}_faiss_index.bin"
            print(f"Faiss index for {modality} is ready with {selected_indices[modality].ntotal} vectors.")
            check_and_create_directory(save_path)
            faiss.write_index(selected_indices[modality], index_save_path)
            print(f"Faiss index for {modality} saved to {index_save_path}")

    if captions_list and wav_paths:
        captions_df = pd.DataFrame({
            'caption': captions_list,
            'wav_path': wav_paths
        })
        captions_csv_path = f"caption_wav_{mode}_path.csv"
        captions_df.to_csv(captions_csv_path, index=False)
        print(f"Captions and wav paths saved to {captions_csv_path}")

    print("Saved all selected indices")


if __name__ == "__main__":
    config = get_config()
    # Load DataLoader in order : WavCaps(AudioCaps-SL,FreeSound, BBC SoundEffects, CLOTHO v2.1)
    # Process embeddings and concatenate into one tensor 48M samples can be processed in GPU memory
    dataloader = pretrain_dataloader(config, bucket=False, is_distributed=False, num_tasks=1, global_rank=0)
    selected_indices, captions_list, wav_paths = generate_faiss_index(config, dataloader)
    save_index(selected_indices, captions_list, wav_paths, config, mode = 'pretrain') # pretrain or train

    # test
    # text_data = ["a dog is barking at a man walking by", "Wind and a man speaking are heard, accompanied by buzzing and ticking."]
    # audio_file = ["./examples/yapping-dog.wav", "./examples/Yb0RFKhbpFJA.flac"]

    # def get_audio_embedding_before_projection(self, data):
    #         """Get the audio embedding from the model

    #         Parameters
    #         ----------
    #         data: a list of dict
    #             the audio input dict list from 'get_audio_feature' method

    #         Returns
    #         ----------
    #         audio_embed: torch.Tensor
    #             a tensor of audio_embeds (N, D)

    #         """
    #         device = next(self.parameters()).device
    #         input_dict = {}
    #         keys = data[0].keys()
    #         for k in keys:
    #             input_dict[k] = torch.cat([d[k].unsqueeze(0) for d in data], dim=0).to(device)
    #         audio_embeds = self.encode_audio(input_dict, device=device)["embedding"] # projection 전을 찾기 위함이다.
    #         return audio_embeds
    
    # clap_model = CLAP_Module(enable_fusion=True)  # 615M
    # clap_model.load_ckpt()
    # # clap_model.model.get_audio_embedding = types.MethodType(get_audio_embedding_before_projection, clap_model.model) 

    # # amodel: str
    # #     audio encoder architecture, default: HTSAT-tiny
    # # tmodel: str
    # #     text encoder architecture, default: roberta
    
    # clap_model.eval()
    # with torch.no_grad():
    #     # text
    #     text_embed = clap_model.get_text_embedding(text_data, use_tensor=True)
    #     print(text_embed)
    #     print(text_embed.shape)

    #     # audio       
    #     audio_embed = clap_model.get_audio_embedding_from_filelist(x=audio_file, use_tensor=True)
        
    #     print(audio_embed)
    #     print(audio_embed.shape)

    # # 생략
    # # import numpy as np

    # # def cosine_similarity(tensor1, tensor2):
    # #     # Normalize each tensor to have unit length
    # #     tensor1_normalized = tensor1 / np.linalg.norm(tensor1)
    # #     tensor2_normalized = tensor2 / np.linalg.norm(tensor2)

    # #     # Compute the cosine similarity
    # #     similarity = np.dot(tensor1_normalized, tensor2_normalized)
    # #     return similarity

    # # Compute similarity between audio and text embeddings
    # # similarities = cosine_similarity(audio_embed[0].cpu(), text_embed[0].cpu())
    # # print(similarities)
    # # similarities = cosine_similarity(audio_embed[0].cpu(), audio_embed[0].cpu())
    # # print(similarities)

    # audio_index = faiss.read_index("./data/original_pretrain_index/audio_faiss_index.bin")
    # text_index = faiss.read_index("./data/original_pretrain_index/text_faiss_index.bin")


    # def load_caption_wav_mapping(csv_path):
    #     df = pd.read_csv(csv_path)
    #     return df['caption'].tolist(), df['wav_path'].tolist()


    # def check_nearest_neighbors(index, queries, k, captions, wav_paths):
    #     # Search the index
    #     D, I = index.search(queries, k)
    #     for i, neighbors in enumerate(I):
    #         print(f"Query {i}:")
    #         for neighbor in neighbors:
    #             print(f" - Neighbor id: {neighbor}, Caption: {captions[neighbor]}, Wav path: {wav_paths[neighbor]}")
    #         print(f" - Distances: {D[i]}")


    # captions_list, wav_paths_list = load_caption_wav_mapping("./data/original_pretrain_index/caption_wav_path.csv")

    # # Convert your embeddings to the correct type for FAISS if they are not already numpy arrays
    # text_query_embeddings = text_embed.cpu().detach().numpy().astype('float32')
    # audio_query_embeddings = audio_embed.cpu().detach().numpy().astype('float32')
    # text_embed

    # k = 16
    # # # text2text
    # # check_nearest_neighbors(text_index, text_query_embeddings, k, captions_list, wav_paths_list)
    # # # text2audio
    # # check_nearest_neighbors(audio_index, text_query_embeddings, k, captions_list, wav_paths_list)
    # # audio2audio
    # check_nearest_neighbors(audio_index, audio_query_embeddings, k, captions_list, wav_paths_list)
    # # audio2text
    # check_nearest_neighbors(text_index, audio_query_embeddings, k, captions_list, wav_paths_list)
    
    
    # # test2 for retrievalIndex class
    # index = RetrievalIndex()
    
    # audio_query_embedding = index.query_embedding(audio_embed)
    # text_query_embedding = index.query_embedding(text_embed)
    
    # index.get_nns("audio", audio_query_embedding, k = 16, show = True)
    # index.get_nns("text", audio_query_embedding, k = 16, show = True)
