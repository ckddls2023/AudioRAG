import json
import torch
import librosa
import random
import numpy as np
import ray
from ray.data import from_items
import torch.nn.functional as F
from BEATs import BEATs, BEATsConfig

ray.init()  # Initialize Ray

json_files = [
  '../data/json_files/AudioSet_SL/as_final.json',
  '../data/json_files/AudioSet/train.json',
  '../data/json_files/Clotho/train.json',
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
        print(waveforms)
        probs = self.model.extract_features(waveforms, padding_mask=padding_mask)[0]
        return probs

def transform_audio(record):
    max_length = 16000*10
    waveform, _ = librosa.load(record["audio"], sr=16000, duration=record["duration"])
    mask = np.ones(max_length, dtype=np.int32)  # 1s for actual data
    if waveform.shape[-1] > max_length:
        max_start = waveform.shape[-1] - max_length
        start = random.randint(0, max_start)
        waveform = waveform[start: start + max_length]
    if waveform.shape[-1] < max_length:
        waveform = np.pad(waveform, (0, max_length - waveform.shape[-1]), mode='constant')
        mask[waveform.shape[-1]:] = 0
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
    
    dataset = from_items(data["data"][:200])
    transformed_dataset = dataset.map(transform_audio)
    result_refs = []
    for i, batch in enumerate(transformed_dataset.iter_batches(batch_size=64)):
        actor = predictor_actors[i % num_gpus]
        result_refs.append(actor.predict.remote(batch))
    result_list = ray.get(result_refs) # Asynchronous execution, we synchronize after all results 
    print(result_list)
    predict_results = torch.cat(result_list, dim=0) # [(B,K), (B,K)...] -> (total_samples, K=527)
    for entry, prediction in zip(data["data"], predict_results):
        top_probs, top_indices = prediction.topk(k=3)
        top_labels = [id2label[str(idx.item())] for idx in top_indices]
        entry["tag"] = ', '.join(top_labels)

    #with open(json_file, 'w') as file:
    #    json.dump(data, file, indent=4)

print("JSON files have been processed and saved with tags.")
