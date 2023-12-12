# Preprocess Audio file and Compute Embeddings
# Build retrieval database : Used for retrieving neighbors
# Build index for similarity search : Train and build a search index for querying neighbors.
import types
from tqdm import tqdm
import argparse
import csv
import sys
from omegaconf import OmegaConf
import faiss
import torch
import pandas as pd
import time
import os
import librosa
import json
import psutil
import numpy as np
import ray
from ray.data import from_items
from laion_clap import CLAP_Module
from laion_clap.training.data import get_audio_features, int16_to_float32, float32_to_int16

# Get the total memory of the system
num_gpus = torch.cuda.device_count()
num_cpus = 4*num_gpus
total_memory = psutil.virtual_memory().total
ray_memory = total_memory*0.8
ray.init(num_cpus=num_cpus, num_gpus=num_gpus, object_store_memory=ray_memory, _memory=ray_memory)

retrieve_json_files = [
  './data/json_files/BBC_Sound_Effects/bbc_final.json',
  './data/json_files/FreeSound/fsd_final.json',
  './data/json_files/SoundBible/sb_final.json',
  './data/json_files/AudioSet_SL/as_final.json',
  './data/json_files/AudioSet/train.json',
  './data/json_files/Clotho/train.json',
]

query_json_files = [
  './data/json_files/AudioSet/train.json',
  './data/json_files/AudioSet/val.json',
  './data/json_files/Clotho/train.json',
  './data/json_files/Clotho/val.json',
]

#index_file_path = "./data/index/index_pretrain.faiss"
index_file_path = "./data/index/index_whole.faiss"
index_exists = os.path.exists(index_file_path)
caption_file_path = "./data/index/big_kb_caption_wav_path.csv"
caption_exists = os.path.exists(caption_file_path)

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

@ray.remote(num_gpus=1)  # Assign one GPU to this actor
class AudioEmbeddingEncoder:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clap = CLAP_Module(enable_fusion=True)  # 615M
        self.clap.load_ckpt()  # download the default pretrained checkpoint.
        self.clap.eval()
        self.clap.to(self.device)

    def encode_audio(self, batch):
        outputs = self.clap.model.get_audio_embedding(batch)
        return outputs
    
    def encode_text(self, batch):
        outputs = self.clap.get_text_embedding(batch)
        return outputs
    
if not caption_exists:
    audio_filenames = []
    audio_captions = []
    for json_file in retrieve_json_files:
        
        with open(json_file, 'r') as file:
            data = json.load(file)
        
        if data["num_captions_per_audio"] > 1:
            for entry in data["data"]:
                entry["caption"] = entry["caption"][0] # Only take first caption as retrieve text
        
        data_filtered = [entry for entry in data["data"] if entry["duration"] <= 40] # Only under 40s
        audio_captions = audio_captions + [entry["caption"] for entry in data_filtered]
        audio_filenames = audio_filenames + [entry["audio"] for entry in data_filtered]
else:
    audio_filenames = []
    audio_captions = []
    print("reload caption index from disk")
    with open(caption_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header
        for row in csv_reader:
            audio_captions.append(row[0])   # Add caption to captions list
            audio_filenames.append(row[1]) # Add file path to file paths list

    
def collate_fn(batch):
    keys = batch.keys()
    batch_length = len(batch["longer"])
    transposed_batch = []
    for i in range(batch_length):
        data_point = {key: torch.tensor(batch[key][i]) for key in keys}
        transposed_batch.append(data_point)
    return transposed_batch
        
total_data_entries = 0
embed_encoder_actors = [AudioEmbeddingEncoder.remote() for _ in range(num_gpus)]
print(f"Available resources : {num_cpus = } {num_gpus =}")
if not index_exists:
    filtered_data = []
    for json_file in retrieve_json_files:
        with open(json_file, 'r') as file:
            data = json.load(file)
        for entry in data["data"]:
            if 'author' in entry:
                del entry["author"]
            if 'description' in entry:
                del entry["description"]
            if 'download_link' in entry:
                del entry["download_link"]
            if 'file_name' in entry:
                del entry["file_name"]
            if 'tags' in entry:
                del entry["tags"]
            if 'href' in entry:
                del entry["href"]
            if 'category' in entry:
                del entry["category"]
            if 'title' in entry:
                del entry["title"]
            del entry["caption"]
        data_filtered = [entry for entry in data["data"] if entry["duration"] <= 40] # Only under 40s
        filtered_data.extend(data_filtered)
    dataset = from_items(filtered_data)
    transformed_dataset = dataset.map(preprocess_waveform)
    result_refs = []
    for i, batch in enumerate(transformed_dataset.iter_batches(batch_size=16, _collate_fn=collate_fn)):
        actor = embed_encoder_actors[i % num_gpus]
        result_refs.append(actor.encode_audio.remote(batch))
    result_list = ray.get(result_refs) # Asynchronous execution, we synchronize after all results 
    audio_embeds = torch.cat(result_list, dim=0)
    audio_embeds = audio_embeds.detach().cpu().numpy()
    dim = audio_embeds.shape[1] # B, H
    #nlist = 512 # 32~512, trade-off between search time, nprobe=32~128
    #quantizer = faiss.IndexFlatL2(dim)
    #index_cpu = faiss.IndexIVFFlat(quantizer, dim, nlist) # Introduce very erronoues results
    #training_samples = np.random.permutation(audio_embeds)[:29000] # Determine the number of training samples, typically ~10% of your dataset, 290k->29k
    #index_cpu.train(training_samples)
    index_cpu = faiss.IndexFlatIP(dim)
    index_cpu.add(audio_embeds)
    faiss.write_index(index_cpu, index_file_path)
else:
    print("reload faiss index from disk")
    index_cpu = faiss.read_index(index_file_path)

# Sanity check
for json_file in retrieve_json_files:
    if os.path.exists(json_file):
        with open(json_file, 'r') as file:
            data = json.load(file)
            data_filtered = [entry for entry in data["data"] if entry["duration"] <= 40] # Only under 40s
            total_data_entries += len(data_filtered)
print(f"length of data samples {total_data_entries} and faiss index embedding {index_cpu.ntotal}, caption {len(audio_captions)}")

@ray.remote
class FaissSearcher:
    def __init__(self, index, nprobe, top_k):
        self.index = index
        self.nprobe = 16
        self.top_k = 5

    def search(self, query_embed):
        distances, indices = self.index.search(query_embed, self.top_k)
        return distances, indices

search_actors = [FaissSearcher.remote(index_cpu, nprobe=16, top_k=5) for _ in range(int(num_cpus))]

# Define a function to split query embeddings into batches
def split_into_batches(embeddings, batch_size):
    return [embeddings[i:i + batch_size] for i in range(0, len(embeddings), batch_size)]

retrieved_results = {}

query_audio_embeds_file = "./data/index/query_audio_embeds.npy"
query_audio_embeds_exists = os.path.exists(query_audio_embeds_file)

query_data = []
tags = []
for json_file in query_json_files:
    with open(json_file, 'r') as file:
        data = json.load(file)
    for entry in data["data"]:
        # Remove specified keys
        keys_to_remove = ['author', 'description', 'download_link', 'file_name', 'href', 'category', 'title', 'caption']
        for key in keys_to_remove:
            if key in entry:
                del entry[key]
        # To find TAG
        tags.append((entry['audio'],entry['tag']))
        for i, tag in enumerate(entry['tag']):
            new_entry = entry.copy()
            new_entry["audio"] = entry["audio"].replace('.wav', f"_{i}.wav")
            query_data.append(new_entry)
        #query_data.append(entry)

if not query_audio_embeds_exists:
    dataset = from_items(query_data)
    transformed_dataset = dataset.map(preprocess_waveform)
    result_refs = []
    for i, batch in enumerate(transformed_dataset.iter_batches(batch_size=16, _collate_fn=collate_fn)):
        actor = embed_encoder_actors[i % num_gpus]
        result_refs.append(actor.encode_audio.remote(batch))
    result_list = ray.get(result_refs) # Asynchronous execution, we synchronize after all results 
    query_audio_embeds = torch.cat(result_list, dim=0)
    query_audio_embeds = query_audio_embeds.detach().cpu().numpy() # B, dim
    np.save(query_audio_embeds_file, query_audio_embeds)
else:
    print("Reload query audio embeds from numpy file")
    query_audio_embeds = np.load(query_audio_embeds_file)

batch_size = 64
query_batches = split_into_batches(query_audio_embeds, 64) # B => N, 64
search_tasks = []
for i, query_batch in enumerate(query_batches):
    actor = search_actors[i % len(search_actors)]
    task = actor.search.remote(query_batch)  # Assuming top_k is defined
    search_tasks.append(task)
search_results = ray.get(search_tasks) # Asynchronous execution, we synchronize after all results 
total_length = sum(len(indices) for (distances, indices) in search_results)
print(f"length of total samples {len(query_data)}, length of indices {total_length}")

#for i, (distances, indices) in enumerate(search_results):
#    for j, indice in enumerate(indices):
#        if query_data[i*batch_size+j]['audio'] == audio_filenames[indice[0]]: # Remove first element which it is itself
#            retrieved_results[query_data[i*batch_size+j]['audio']] = [(audio_filenames[i],audio_captions[i]) for i in indice[1:]]
#        else:
#            retrieved_results[query_data[i*batch_size+j]['audio']] = [(audio_filenames[i],audio_captions[i]) for i in indice]

# To find tags 
distances = [elem[0] for elem in search_results]
indices = [elem[1] for elem in search_results]
distances = np.concatenate(distances) # B, 5
indices = np.concatenate(indices)
start_idx = 0
text_embed_tags = []
for i, (audio, tag) in enumerate(tags):
    actor = embed_encoder_actors[i % num_gpus]
    texts = ["This is sound of " + caption for caption in tag for _ in range(5)] # top_5
    text_embed_tags.append(actor.encode_text.remote(texts))

for i, (audio, tag) in enumerate(tags):
    end_idx = start_idx + len(tag)
    distance = distances[start_idx:end_idx].reshape(-1)
    indice = indices[start_idx:end_idx].reshape(-1)
    text_embed_tag = ray.get(text_embed_tags[i]) # synchronize, 5*len(tag), dim
    audio_embed_tag = np.array([index_cpu.reconstruct(int(id)) for id in indice]) # 5*len(tag), dim
    weights = np.einsum('ij,ij->i', text_embed_tag, audio_embed_tag)
    weighted_distance = distance / (weights + 1e-6) # if similar, we use di
    sorted_indices = np.argsort(distance)
    sorted_indice = indice[sorted_indices]
    selected_indice = sorted_indice[:5]
    retrieved_results[audio] = [(audio_filenames[i],audio_captions[i]) for i in selected_indice]
    start_idx = end_idx

# Save the results
with open('retrieved_results.json', 'w') as f:
    json.dump(retrieved_results, f, indent=4)
