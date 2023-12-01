import json
import torch
import librosa
import random
import numpy as np
import psutil
import ray
from ray.data import from_items
import torch.nn.functional as F
from BEATs import BEATs, BEATsConfig
from models.audiosep import AudioSep
from utils import get_ss_model
from pipeline import inference


# Get the total memory of the system
total_memory = psutil.virtual_memory().total
reserved_memory = total_memory * 0.1
ray_memory = total_memory - reserved_memory
ray.init(object_store_memory=ray_memory, _memory=ray_memory)

json_files = [
  '../data/json_files/AudioSet/val.json',
  '../data/json_files/Clotho/train.json',
  '../data/json_files/Clotho/val.json',
]
  

@ray.remote(num_gpus=1)  # Assign one GPU to this actor
class AudioSetTagPredictor:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path)
        cfg = BEATsConfig(checkpoint['cfg'])
        self.model = BEATs(cfg)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        self.model.to(self.device)

    def predict(self, batch):
        batch_waveforms = batch['waveform'].copy()
        batch_masks = batch['mask'].copy()
        waveforms = torch.from_numpy(batch_waveforms).to(self.device)
        padding_mask = torch.from_numpy(batch_masks).bool().to(self.device)
        probs = self.model.extract_features(waveforms, padding_mask=padding_mask)[0]
        return probs

def transform_audio(record):
    max_length = 16000*10
    waveform, _ = librosa.load(record["audio"], sr=16000, duration=record["duration"], mono=True)
    mask = np.zeros(max_length, dtype=np.int32)  # 1s for actual data
    if waveform.shape[-1] > max_length:
        max_start = waveform.shape[-1] - max_length
        start = random.randint(0, max_start)
        waveform = waveform[start: start + max_length]
    if waveform.shape[-1] < max_length:
        waveform = np.pad(waveform, (0, max_length - waveform.shape[-1]), mode='constant')
        mask[waveform.shape[-1]:] = 1
    return {'waveform': waveform, 'mask': mask}

with open("./audioset-id2label.json", 'r') as file:
    id2label = json.load(file)
    
def get_gpu_count():
    return torch.cuda.device_count()

num_gpus = get_gpu_count()
predictor_actors = [AudioSetTagPredictor.remote('./BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt') for _ in range(num_gpus)]


for json_file in json_files:
    
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    dataset = from_items(data["data"])
    transformed_dataset = dataset.map(transform_audio)
    result_refs = []
    for i, batch in enumerate(transformed_dataset.iter_batches(batch_size=16)):
        actor = predictor_actors[i % num_gpus]
        result_refs.append(actor.predict.remote(batch))
    result_list = ray.get(result_refs) # Asynchronous execution, we synchronize after all results 
    predict_results = torch.cat(result_list, dim=0) # [(B,K), (B,K)...] -> (total_samples, K=527)
    for entry, prediction in zip(data["data"], predict_results):
        top_probs, top_indices = prediction.topk(k=3)
        top_labels = [id2label[str(idx.item())] for idx in top_indices]
        entry["tag"] = top_labels # Save it as list

    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)

print("JSON files have been processed and saved with tags.")

# Terminate the AudioSetTagPredictor actors
for actor in predictor_actors:
    ray.kill(actor)

@ray.remote(num_gpus=1)
class AudioSeparator:
    def __init__(self, model_name, config_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ss_model = get_ss_model(config_path)
        self.model = AudioSep.from_pretrained(model_name, ss_model=ss_model)
        self.model.to(self.device)

    def separate(self, audio_file, text, output_file):
        # AudioSep processes the audio at 32 kHz sampling rate
        inference(self.model, audio_file, text, output_file, self.device)

separator_actors = [AudioSeparator.remote("nielsr/audiosep-demo", 'config/audiosep_base.yaml') for _ in range(num_gpus)]


json_files = ['../data/json_files/AudioSet/train.json'] + json_files
for json_file in json_files:
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Iterate over each entry and separate audio based on the tag
    separation_refs = []
    for entry in data["data"]:
        audio_file = entry['audio']
        tags = entry['tag']
        for i, tag in enumerate(tags):
            output_file = audio_file.replace(".wav", f"_{i}.wav")
            actor = separator_actors[i % num_gpus]
            separation_refs.append(actor.separate.remote(audio_file, tag, output_file))

    ray.get(separation_refs)

print("Audio files have been processed and separated based on tags.")
