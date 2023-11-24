import json
import torch
import librosa
import numpy as np
import ray
from ray.data import from_items
import torch.nn.functional as F
from BEATs import BEATs, BEATsConfig

ray.init()  # Initialize Ray

json_files = [
  'data/json_files/BBC_Sound_Effects/bbc_final.json',
  'data/json_files/FreeSound/fsd_final.json',
  'data/json_files/SoundBible/sb_final.json',
  'data/json_files/AudioSet_SL/as_final.json',
  'data/json_files/AudioSet/train.json',
  'data/json_files/Clotho/train.json',
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
        wav_list = [record["audio_data"] for record in batch]
        max_length = max(len(wav) for wav in wav_list)
        padded_wav_list = []
        masks = []
        for wav in wav_list:
            padded_wav = F.pad(torch.tensor(wav), (0, max_length - len(wav)))
            padded_wav_list.append(padded_wav)
            mask = torch.tensor([1] * len(wav) + [0] * (max_length - len(wav)))
            masks.append(mask)
        waveforms = torch.stack(padded_wav_list, dim=0)
        padding_mask = torch.stack(masks, dim=0)
        waveforms = waveforms.to(self.device)
        padding_mask = padding_mask.to(self.device)
        probs = self.model.extract_features(waveforms, padding_mask=padding_mask)[0]
        return probs

def transform_audio(record):
    audio, _ = librosa.load(record["wav_path"], sr=16000, duration=record["duration"])
    record["audio_data"] = audio
    return record

with open("./audioset-id2label.json", 'r') as file:
    id2label = json.load(file)

# Processing each JSON file
for json_file in json_files:
    
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    dataset = from_items(data["data"])
    transformed_dataset = dataset.map(transform_audio)
    predictor_actor = AudioSetTagPredictor.remote('./BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt')
    result_dataset = transformed_dataset.map_batches(
        lambda batch: ray.get(predictor_actor.predict.remote(batch)),
        batch_format="list",  # since your data is a list of dictionaries
        num_gpus=1, 
        batch_size=64
        )
    result_list = result_dataset.take_all()
    predict_results = torch.cat(result_list, dim=0) # (B, K=527)
    for entry, prediction in zip(data["data"], predict_results):
        top_probs, top_indices = prediction.topk(k=3)
        top_labels = [id2label[idx.item()] for idx in top_indices]
        entry["tag"] = ', '.join(top_labels)

    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)

print("JSON files have been processed and saved with tags.")
