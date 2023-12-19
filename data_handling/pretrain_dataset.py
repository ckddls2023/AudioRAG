#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import json
import random
import librosa
import torch
import ruamel.yaml as yaml
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
        with open(file, "r") as f:
            json_obj = json.load(f)
            for item in json_obj["data"]:
                if "FreeSound" in file and blacklist:
                    if item["id"] in blacklist["FreeSound"]:
                        continue
                elif "AudioSet" in file and blacklist:
                    if item["id"] in blacklist["AudioSet"]:
                        continue
                if item["duration"] > 40.0:  # Avoid too much long audios
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

        self.sr = audio_config["sr"]
        if audio_config["max_length"] != 0:
            self.max_length = audio_config["max_length"] * self.sr
        else:
            self.max_length = 0
            
        self.audio_cfg = {
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

    def __len__(self):
        return len(self.json_data)
    
    def preprocess_waveform(self, wav_path, duration):
        waveform, sr = librosa.load(wav_path, sr=self.sr, duration=duration)
        audio_waveform = int16_to_float32(float32_to_int16(waveform))
        audio_waveform = torch.from_numpy(audio_waveform).float()
        temp_dict = {}
        temp_dict = get_audio_features(
            temp_dict, audio_waveform, 480000,
            data_truncating='fusion',
            data_filling='repeatpad',
            audio_cfg=self.audio_cfg,
            require_grad=False,
        )
        return temp_dict

    def __getitem__(self, index):
        item = self.json_data[index]
        wav_path = item["audio"]
        duration = item["duration"]
        audio_feature = self.preprocess_waveform(wav_path, duration)
        caption = item["caption"]
        if self.train and isinstance(caption, list):
            caption = random.choice(item["caption"])
        caption = text_preprocess(caption)
        retr_audio_features = []
        retr_captions = []
        if wav_path in self.retrieve_map and self.top_k > 0:
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
            retr_captions = [text_preprocess(caption) for (retr_wav_path, caption) in selected_items]
        return audio_feature, caption, wav_path, retr_audio_features, retr_captions


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
    dataset = AudioLanguagePretrainDataset(config[subset], config["audio_args"], blacklist, 'train' in subset, retrieve_map, top_k)
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
        drop_last=True,
        collate_fn=collate_fn,
    )


if __name__ == '__main__':
    with open("../../WavCaps/captioning/settings/pretrain.yaml", "r") as f:
        config = yaml.safe_load(f)
    print(config)
