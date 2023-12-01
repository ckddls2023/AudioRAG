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
import json
import numpy as np
import ray
from ray.data import from_items
from laion_clap import CLAP_Module
from laion_clap.training.data import get_audio_features, int16_to_float32, float32_to_int16

ray.init()  # Initialize Ray

retrieve_json_files = [
  './data/json_files/BBC_Sound_Effects/bbc_final.json',
  './data/json_files/FreeSound/fsd_final.json',
  './data/json_files/SoundBible/sb_final.json',
  './data/json_files/AudioSet_SL/as_final.json',
]

query_json_files = [
  './data/json_files/AudioSet/train.json',
  './data/json_files/AudioSet/val.json',
  './data/json_files/Clotho/train.json',
  './data/json_files/Clotho/val.json',
]

index_file_path = "./index_final.faiss"
index_exists = os.path.exists(index_file_path)

def preprocess_waveform(record):
    audio_cfg = {
        "audio_length": 1024,
        "clip_samples": 480000,
        "mel_bins": 64,
        "sample_rate": 48000,
        "window_size": 1024,
        "hop_size": 480,
        "fmin": 50,
        "fmax": 14000,
        "class_num": 527,
        "model_type": "HTSAT",
        "model_name": "base"
    }
    waveform, sr = librosa.load(record["audio"], sr=48000, duration=record["duration"])
    audio_waveform = int16_to_float32(float32_to_int16(waveform))
    audio_waveform = torch.from_numpy(audio_waveform).float()
    temp_dict = {}
    temp_dict = get_audio_features(
        temp_dict, audio_waveform, 480000,
        data_truncating='fusion',
        data_filling='repeatpad',
        audio_cfg=audio_cfg,
        require_grad=False,
    )
    return temp_dict

def get_gpu_count():
    return torch.cuda.device_count()

@ray.remote(num_gpus=1)  # Assign one GPU to this actor
class AudioEmbeddingEncoder:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clap = CLAP_Module(enable_fusion=True)  # 615M
        self.clap.load_ckpt()  # download the default pretrained checkpoint.
        self.clap.eval()
        self.clap.to(self.device)

    def encode(self, batch):
        outputs = self.clap.model.get_audio_embedding(batch)
        return outputs
    
audio_filenames = []
audio_captions = []
for json_file in retrieve_json_files:
    
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    audio_captions = audio_captions + [entry["caption"] for entry in data["data"]]
    audio_filenames = audio_filenames + [entry["audio"] for entry in data["data"]]
    
def collate_fn(batch):
    keys = batch.keys()
    batch_length = len(batch["longer"])
    transposed_batch = []
    for i in range(batch_length):
        data_point = {key: torch.tensor(batch[key][i]) for key in keys}
        transposed_batch.append(data_point)
    return transposed_batch
        
if not index_exists:
    num_gpus = get_gpu_count()
    embed_encoder_actors = [AudioEmbeddingEncoder.remote() for _ in range(num_gpus)]
    audio_embeds = []

    for json_file in retrieve_json_files:
        
        with open(json_file, 'r') as file:
            data = json.load(file)
            
        dataset = from_items(data["data"])
        transformed_dataset = dataset.map(preprocess_waveform)
        result_refs = []
        for i, batch in enumerate(transformed_dataset.iter_batches(batch_size=32, _collate_fn=collate_fn)):
            actor = embed_encoder_actors[i % num_gpus]
            result_refs.append(actor.encode.remote(batch))
        result_list = ray.get(result_refs) # Asynchronous execution, we synchronize after all results 
        audio_embeds = audio_embeds + [torch.cat(result_list, dim=0)] # [(B,512), (B,512)...] -> (total_samples, 512)
    
    audio_embeds = torch.cat(audio_embeds, dim=0)
    audio_embeds = audio_embeds.cpu().numpy()

    dim = audio_embeds.shape[1] # B, H
    nlist = 512 # 32~512, trade-off between search time, nprobe=32~128
    quantizer = faiss.IndexFlatL2(dim)
    index_cpu = faiss.IndexIVFFlat(quantizer, dim, nlist)
    training_samples = np.random.permutation(audio_embeds)[:29000] # Determine the number of training samples, typically ~10% of your dataset, 290k->29k
    index_cpu.train(training_samples)
    index_cpu.add(audio_embeds)
    faiss.write_index(index_cpu, index_file_path)
else:
    index_cpu = faiss.read_index(index_file_path)

# Sanity check
total_data_entries = 0
for json_file in retrieve_json_files:
    if os.path.exists(json_file):
        with open(json_file, 'r') as file:
            data = json.load(file)
            total_data_entries += len(data["data"])
print(f"length of data samples {total_data_entries} and faiss index embedding {index_cpu.ntotal}")
    

@ray.remote
class FaissSearcher:
    def __init__(self, index, nprobe, top_k):
        self.index = index
        self.nprobe = 16
        self.top_k = 5

    def search(self, query_embed):
        distances, indices = self.index.search(query_embed, self.top_k)
        return distances, indices

num_cpus = ray.available_resources()["CPU"]
search_actors = [FaissSearcher.remote(index_cpu, nprobe=16, top_k=5) for _ in range(int(num_cpus))]

# Define a function to split query embeddings into batches
def split_into_batches(embeddings, batch_size):
    return [embeddings[i:i + batch_size] for i in range(0, len(embeddings), batch_size)]

retrieved_results = {}

for json_file in query_json_files:
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    dataset = from_items(data["data"])
    transformed_dataset = dataset.map(preprocess_waveform)
    result_refs = []
    for i, batch in enumerate(transformed_dataset.iter_batches(batch_size=32, _collate_fn=collate_fn)):
        actor = embed_encoder_actors[i % num_gpus]
        result_refs.append(actor.encode.remote(batch))
    result_list = ray.get(result_refs) # Asynchronous execution, we synchronize after all results 
    query_audio_embeds = torch.cat(result_list, dim=0)
    query_audio_embeds = query_audio_embeds.cpu().numpy() # B, dim
    
    batch_size = 64
    query_batches = split_into_batches(query_audio_embeds, 64) # B => N, 64
    search_tasks = []
    for i, query_batch in enumerate(query_batches):
        actor = search_actors[i % len(search_actors)]
        task = actor.search.remote(query_batch)  # Assuming top_k is defined
        search_tasks.append(task)
        
    data_batches = split_into_batches(dataset, 64)  # B => N, 64
    for data_batch, task in zip(data_batches, search_tasks):
        distances, indices = ray.get(task)
        for entry, indice in zip(data_batch, indices):
            retrieved_results[entry['audio']] = [(audio_filenames[i],audio_captions[i]) for i in indice]

# Save the results
with open('retrieved_results.json', 'w') as f:
    json.dump(retrieved_results, f)
