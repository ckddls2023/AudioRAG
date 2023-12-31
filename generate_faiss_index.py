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
from languagebind import LanguageBindAudio, LanguageBindAudioTokenizer, LanguageBindAudioProcessor, LanguageBindAudioConfig

num_gpus = torch.cuda.device_count()
num_cpus = 8*num_gpus
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

# query_json_files = [
#   './data/json_files/BBC_Sound_Effects/bbc_final.json',
#   './data/json_files/FreeSound/fsd_final.json',
#   './data/json_files/SoundBible/sb_final.json',
#   './data/json_files/AudioSet_SL/as_final.json',
#   './data/json_files/AudioSet/train.json',
#   './data/json_files/Clotho/train.json',
# ]

query_json_files = [
  './data/json_files/AudioSet/train.json',
  './data/json_files/AudioSet/val.json',
  './data/json_files/Clotho/train.json',
  './data/json_files/Clotho/val.json',
]

audio_index_file_path = "./data/index/audio_index.faiss"
text_index_file_path = "./data/index/text_index.faiss"
index_exists = os.path.exists(audio_index_file_path)
caption_file_path = "./data/index/big_kb_caption_wav_path.csv"
caption_exists = os.path.exists(caption_file_path)

@ray.remote
class PreprocessingActor:
    def __init__(self):
        config = LanguageBindAudioConfig.from_pretrained('./checkpoint')
        tokenizer = LanguageBindAudioTokenizer.from_pretrained('./checkpoint/')
        self.audio_process = LanguageBindAudioProcessor(config, tokenizer)

    def prepare_batch(self, batch):
        audios = batch['audio'].tolist()
        captions = batch['caption'].tolist()
        batch = self.audio_process(audios, captions, return_tensors='pt')
        return batch


@ray.remote(num_gpus=1)  # Assign one GPU to this actor
class AudioEmbeddingEncoder:
    def __init__(self, model_ref):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_ref.to(self.device)

    def encode_audio(self, batch):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            out = self.model(**batch)
        return out.image_embeds.detach().cpu()
    
    def encode_text(self, batch):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            out = self.model(**batch)
        return out.text_embeds.detach().cpu()
        
    def encode_audio_text(self, batch):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            out = self.model(**batch)
        return {'image_embeds': out.image_embeds.detach().cpu(), 'text_embeds': out.text_embeds.detach().cpu()}
    
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
    # Writing to the CSV file
    with open(caption_file_path, mode='w', newline='', encoding='utf-8') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["Caption", "Audio Filename"])  # Writing the header
        for caption, filename in zip(audio_captions, audio_filenames):
            csv_writer.writerow([caption, filename])
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

            
total_data_entries = 0
embed_encoder_actors = [AudioEmbeddingEncoder.remote(ray.put(LanguageBindAudio.from_pretrained('./checkpoint/'))) for _ in range(num_gpus)]
preprocessing_actors = [PreprocessingActor.remote() for _ in range(num_cpus)]

max_in_flight_tasks = 100  # Adjust this number based on your system's capability
print(f"Available resources : {num_cpus = } {num_gpus =}")
if not index_exists:
    filtered_data = []
    for json_file in retrieve_json_files:
        with open(json_file, 'r') as file:
            data = json.load(file)
        for entry in data["data"]:
            keys_to_remove = ['author', 'description', 'category', 'download_link', 'file_name', 'href', 'category', 'title', 'tag', 'tags']
            for key in keys_to_remove:
                if key in entry:
                    del entry[key]
            if data["num_captions_per_audio"] > 1:
                entry["caption"] = entry["caption"][0] # Only take first caption as retrieve text and encode
        data_filtered = [entry for entry in data["data"] if entry["duration"] <= 40] # Only under 40s
        filtered_data.extend(data_filtered)
    dataset = from_items(filtered_data)
    result_refs = []
    results = []
    total_batch = len(filtered_data) // 24
    for i, batch in enumerate(dataset.iter_batches(batch_size=24)):
        if i % 100 == 0:  # Print progress every 10 iterations
            print(f"Processing batch {i+1}/{total_batch}...")
        if len(result_refs) >= max_in_flight_tasks:
            done_refs, result_refs = ray.wait(result_refs, num_returns=len(result_refs) - max_in_flight_tasks + 1)
            results.extend(ray.get(done_refs))
            for ref in done_refs:
                del ref
        preprocessing_actor = preprocessing_actors[i % num_cpus]
        preprocessed_ref = preprocessing_actor.prepare_batch.remote(batch)
        actor = embed_encoder_actors[i % num_gpus]
        result_refs.append(actor.encode_audio_text.remote(preprocessed_ref))
    results.extend(ray.get(result_refs)) # Asynchronous execution, we synchronize after all results 

    audio_embeds = [out['image_embeds'] for out in results]
    audio_embeds = torch.cat(audio_embeds, dim=0)
    audio_embeds = audio_embeds.detach().cpu().contiguous().numpy()
    audio_dim = audio_embeds.shape[1] # B, H
    text_embeds = [out['text_embeds'] for out in results]
    text_embeds = torch.cat(text_embeds, dim=0)
    text_embeds = text_embeds.detach().cpu().contiguous().numpy()
    text_dim = text_embeds.shape[1] # B, H
    #nlist = 512 # 32~512, trade-off between search time, nprobe=32~128
    #quantizer = faiss.IndexFlatL2(dim)
    #index_cpu = faiss.IndexIVFFlat(quantizer, dim, nlist) # Introduce very erronoues results
    #training_samples = np.random.permutation(audio_embeds)[:29000] # Determine the number of training samples, typically ~10% of your dataset, 290k->29k
    #index_cpu.train(training_samples)
    audio_index_cpu = faiss.IndexFlatIP(audio_dim)
    audio_index_cpu.add(audio_embeds)
    np.save("./data/index/audio_embeds.npy", audio_embeds)
    faiss.write_index(audio_index_cpu, audio_index_file_path)
    text_index_cpu = faiss.IndexFlatIP(text_dim)
    text_index_cpu.add(text_embeds)
    np.save("./data/index/text_embeds.npy", text_embeds)
    faiss.write_index(text_index_cpu, text_index_file_path)
else:
    print("reload faiss index from disk")
    audio_index_cpu = faiss.read_index(audio_index_file_path)
    text_index_cpu = faiss.read_index(text_index_file_path)

# Sanity check
for json_file in retrieve_json_files:
    if os.path.exists(json_file):
        with open(json_file, 'r') as file:
            data = json.load(file)
            data_filtered = [entry for entry in data["data"] if entry["duration"] <= 40] # Only under 40s
            total_data_entries += len(data_filtered)
print(f"length of data samples {total_data_entries} and faiss index embedding {audio_index_cpu.ntotal}, caption {len(audio_captions)}")

@ray.remote
class FaissSearcher:
    def __init__(self, index, nprobe, top_k):
        self.index = index
        self.nprobe = 16
        self.top_k = 5

    def search(self, query_embed):
        distances, indices = self.index.search(query_embed, self.top_k)
        return distances, indices

search_actors = [FaissSearcher.remote(text_index_cpu, nprobe=16, top_k=5) for _ in range(int(num_cpus))]

# Define a function to split query embeddings into batches
def split_into_batches(embeddings, batch_size):
    return [embeddings[i:i + batch_size] for i in range(0, len(embeddings), batch_size)]

retrieved_results = {}

query_audio_embeds_file = "./data/index/query_audio_embeds.npy"
query_audio_embeds_exists = os.path.exists(query_audio_embeds_file)

query_data = []
# tags = []
for json_file in query_json_files:
    with open(json_file, 'r') as file:
        data = json.load(file)
    for entry in data["data"]:
        # Remove specified keys
        keys_to_remove = ['author', 'description', 'download_link', 'category', 'file_name', 'href', 'category', 'title']
        for key in keys_to_remove:
            if key in entry:
                del entry[key]
        if data["num_captions_per_audio"] > 1:
            entry["caption"] = entry["caption"][0] # Only take first caption as retrieve text and encode
            
        # To find TAG
        # tags.append((entry['audio'],entry['tag']))
        # for i, tag in enumerate(entry['tag']):
        #     new_entry = entry.copy()
        #     new_entry["audio"] = entry["audio"].replace('.wav', f"_{i}.wav")
        #     query_data.append(new_entry)
    query_data.extend([entry for entry in data["data"] if entry["duration"] <= 40])

if not query_audio_embeds_exists:
    dataset = from_items(query_data)
    results = []
    result_refs = []
    total_batch = len(query_data) // 24
    for i, batch in enumerate(dataset.iter_batches(batch_size=24)):
        if i % 100 == 0:  # Print progress every 10 iterations
            print(f"Processing batch {i+1}/{total_batch}...")
        if len(result_refs) >= max_in_flight_tasks:
            done_refs, result_refs = ray.wait(result_refs, num_returns=len(result_refs) - max_in_flight_tasks + 1)
            results.extend(ray.get(done_refs))
            for ref in done_refs:
                del ref
        preprocessing_actor = preprocessing_actors[i % num_cpus]
        preprocessed_ref = preprocessing_actor.prepare_batch.remote(batch)
        actor = embed_encoder_actors[i % num_gpus]
        result_refs.append(actor.encode_audio.remote(preprocessed_ref))
    results.extend(ray.get(result_refs)) # Asynchronous execution, we synchronize after all results 
    query_audio_embeds = torch.cat(results, dim=0)
    query_audio_embeds = query_audio_embeds.detach().cpu().contiguous().numpy() # B, dim
    np.save(query_audio_embeds_file, query_audio_embeds)
else:
    print("Reload query audio embeds from numpy file")
    query_audio_embeds = np.load(query_audio_embeds_file)

batch_size = 1 # Due to unstable search results
query_batches = split_into_batches(query_audio_embeds, batch_size) # B => N, 64
search_tasks = []
for i, query_batch in enumerate(query_batches):
    actor = search_actors[i % len(search_actors)]
    task = actor.search.remote(query_batch)  # Assuming top_k is defined
    search_tasks.append(task)
search_results = ray.get(search_tasks) # Asynchronous execution, we synchronize after all results 
total_length = sum(len(indices) for (distances, indices) in search_results)
print(f"length of total samples {len(query_data)}, length of indices {total_length}")

for i, (distances, indices) in enumerate(search_results):
   for j, indice in enumerate(indices):
       print(distances[j])
       if query_data[i*batch_size+j]['audio'] == audio_filenames[indice[0]]: # Remove first element which it is itself
           retrieved_results[query_data[i*batch_size+j]['audio']] = [(audio_filenames[idx],audio_captions[idx]) for idx in indice[1:]]
       else:
           retrieved_results[query_data[i*batch_size+j]['audio']] = [(audio_filenames[idx],audio_captions[idx]) for idx in indice]

# To find tags 
# distances = [elem[0] for elem in search_results]
# indices = [elem[1] for elem in search_results]
# distances = np.concatenate(distances) # B, 5
# indices = np.concatenate(indices)
# start_idx = 0
# text_embed_tags = []
# for i, (audio, tag) in enumerate(tags[:16]):
#     actor = embed_encoder_actors[i % num_gpus]
#     texts = ["a sound of " + caption for caption in tag for _ in range(5)] # top_5
#     text_embed_tags.append(actor.encode_text.remote(texts))

# for i, (audio, tag) in enumerate(tags[:16]):
#     end_idx = start_idx + len(tag)
#     distance = distances[start_idx:end_idx].reshape(-1)
#     indice = indices[start_idx:end_idx].reshape(-1)
#     text_embed_tag = ray.get(text_embed_tags[i]) # synchronize, 5*len(tag), dim
#     audio_embed_tag = np.array([audio_index_cpu.reconstruct(int(id)) for id in indice]) # 5*len(tag), dim
#     weights = np.einsum('ij,ij->i', text_embed_tag, audio_embed_tag)
#     weighted_distance = distance / (np.abs(weights) + 1e-6) # if similar, we use di
#     sorted_indices = np.argsort(np.abs(distance))
#     sorted_indice = indice[sorted_indices]
#     selected_indice = sorted_indice[:5]
#     retrieved_results[audio] = [(audio_filenames[i],audio_captions[i]) for i in selected_indice]
#     start_idx = end_idx

# Save the results
with open('retrieved_results.json', 'w') as f:
    json.dump(retrieved_results, f, indent=4)
