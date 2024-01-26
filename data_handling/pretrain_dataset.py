import json
import random
import librosa
import soundfile as sf
import torch
import random
import os
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from data_handling.datamodule import collate_fn
from data_handling.sampler import BySequenceLengthSampler, BySequenceBatchSampler
from data_handling.text_transform import text_preprocess
from laion_clap.training.data import get_audio_features, int16_to_float32, float32_to_int16


def load_json_file(files, blacklist=None, train=True):
    json_data = []
    audio_id = 0
    if blacklist:
        with open(blacklist, 'r') as f:
            blacklist = json.load(f)
    for file in files:
        parent_path = os.path.basename(os.path.dirname(file))  # Extracts the parent directory name
        with open(file, "r") as f:
            json_obj = json.load(f)
            for i, item in enumerate(random.sample(json_obj["data"], min(len(json_obj["data"]),500000))):
                item["embedding_path"] = f"./data/embeddings/{parent_path}/{i:07d}.npy"
                if "FreeSound" in file and blacklist:
                    if item["id"] in blacklist["FreeSound"]:
                        continue
                elif "AudioSet" in file and blacklist:
                    if item["id"] in blacklist["AudioSet"]:
                        continue
                if item["duration"] > 40.0 or item["duration"] < 5.0:  # Avoid too much short or long audios
                    continue
                json_data.append(item)
                audio_id += 1
    return json_data


class AudioLanguagePretrainDataset(Dataset):

    def __init__(self, json_files, audio_config, blacklist=None, train=True, retrieve_map="", top_k=2):

        self.json_data = load_json_file(json_files, blacklist, train)
        self.train = train
        self.lengths = [item["duration"] for item in self.json_data]
        self.top_k = top_k
        self.retrieve_map = {}
        self.noisy_k = 0
        if retrieve_map:
            with open(retrieve_map, 'r') as file:
                self.retrieve_map = json.load(file)

        self.audio_cfg = OmegaConf.to_container(audio_config, resolve=True)
        self.sr = self.audio_cfg["sample_rate"]
            

    def __len__(self):
        return len(self.json_data)
    
    def read_wav(self, wav_path):
        wav, sr = sf.read(wav_path)
        if len(wav.shape) == 2:
            wav = wav[:, 0]
        if len(wav) > 30 * sr:
            wav = wav[: 30 * sr]
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000, res_type="fft")
        return wav
    
    def preprocess_waveform(self, wav_path, duration, fuse=True):
        waveform, sr = librosa.load(wav_path, sr=self.sr, duration=duration)
        audio_waveform = int16_to_float32(float32_to_int16(waveform))
        max_length = self.audio_cfg["clip_samples"]
        if not fuse and len(audio_waveform) > max_length: # Seperate into partial segments
            results = []
            segments = [audio_waveform[i:i+max_length] for i in range(0, len(audio_waveform), max_length)]
            for segment in segments:
                segment_tensor = torch.from_numpy(segment).float()
                temp_dict = {}
                temp_dict = get_audio_features(
                    temp_dict, segment_tensor, max_length,
                    data_truncating='rand_trunc',
                    data_filling='repeatpad',
                    audio_cfg=self.audio_cfg,
                    require_grad=False,
                )
                results.append(temp_dict)
            return results
        else: # Fused, single waveform
            audio_waveform = torch.from_numpy(audio_waveform).float()
            temp_dict = {}
            temp_dict = get_audio_features(
                temp_dict, audio_waveform, max_length,
                data_truncating='rand_trunc',
                data_filling='repeatpad',
                audio_cfg=self.audio_cfg,
                require_grad=False,
            )
            return temp_dict

    def __getitem__(self, index):
        item = self.json_data[index]
        wav_path = item["audio"]
        embedding_path = item["embedding_path"]
        # embedding = np.load(embedding_path)
        duration = item["duration"]
        audio_feature = self.preprocess_waveform(wav_path, duration, fuse=True) # Always not fuse it's feature
        # audio_feature = self.read_wav(wav_path) # Always not fuse it's feature
        caption = item["caption"]
        if self.train and isinstance(caption, list):
            caption = random.choice(item["caption"])
        caption = text_preprocess(caption)
        retr_audio_features = []
        retr_captions = []
        if wav_path in self.retrieve_map:
            retrieve_items = self.retrieve_map[wav_path]
            weights = list(range(len(retrieve_items), 0, -1))
            if self.train:
                selected_items = random.choices(retrieve_items, weights=weights, k=self.top_k)
            else:
                selected_items = retrieve_items[:self.top_k]
            # for i in range(self.noisy_k): # Add noisy examples
            #     other_key = random.choice([k for k in self.retrieve_map.keys() if k != wav_path])
            #     random_item = random.choice(self.retrieve_map[other_key])
            #     selected_items[i] = random_item
            retr_audio_features = [self.preprocess_waveform(retr_wav_path, duration) for (retr_wav_path, caption) in selected_items]
            # retr_audio_features = [self.read_wav(retr_wav_path) for (retr_wav_path, caption) in selected_items]
            retr_captions = [text_preprocess(caption) for (retr_wav_path, caption) in selected_items]
        return audio_feature, caption, wav_path, retr_audio_features, retr_captions, embedding_path


def pretrain_dataloader(config,
                        subset: str = "train_jsons",
                        bucket: bool = True,
                        bucket_boundaries: tuple = (5, 30, 6),
                        is_distributed: bool = False,
                        num_tasks: int = 0,
                        global_rank: int = 0,
                        retrieve_map="",
                        top_k=2,
                        shuffle=False):
    blacklist = None if 'val' in subset else config.blacklist
    batch_size = 1 if 'val' in subset else config.data_args.batch_size
    dataset = AudioLanguagePretrainDataset(config[subset], config.audio_args, blacklist, 'train' in subset, retrieve_map, top_k)
    if bucket:
        sampler = BySequenceLengthSampler(lengths=dataset.lengths,
                                          bucket_boundaries=bucket_boundaries,
                                          batch_size=batch_size,
                                          drop_last=True,
                                          seed=config["seed"])
        return DataLoader(dataset=dataset,
                          batch_sampler=BySequenceBatchSampler(sampler, batch_size=batch_size, drop_last=False),
                          shuffle=shuffle,
                          pin_memory=True,
                          drop_last=True,
                          num_workers=config["data_args"]["num_workers"],
                          collate_fn=collate_fn)
    elif is_distributed:
        sampler = DistributedSampler(dataset,
                                     num_replicas=num_tasks,
                                     rank=global_rank,
                                     drop_last=False,
                                     shuffle=True)
    else:
        sampler = None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=config["data_args"]["num_workers"],
        pin_memory=True,
        sampler=sampler,
        shuffle=shuffle,
        drop_last=False,
        collate_fn=collate_fn,
    )

